"""Fixed-price basic flipbooks: isolated SQLite, no real clients or paid API."""
import asyncio
import hashlib
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import database
import credits
import holiday_credits as gifts
import tool_credits as tools


def fingerprint(name):return hashlib.sha256(name.encode()).hexdigest()
def operation():return str(uuid.uuid4())


class FlipbookCreditsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='flipbook-billing-')
        self.old_path=database.DB_PATH;database.DB_PATH=str(Path(self.tmp.name)/'test.db')
        await database.init_db()
        for user in (1,2):
            await self.sql('INSERT INTO users(id,email,password_hash) VALUES(?,?,?)',(user,f'flipbook{user}@example.invalid','unused'))
            await self.sql("INSERT INTO credits(user_id,balance,monthly_quota,plan_name,sub_status) VALUES(?,3,3,'free','none')",(user,))
            await self.sql("INSERT INTO credit_transactions(user_id,delta,reason) VALUES(?,3,'signup_bonus')",(user,))

    async def asyncTearDown(self):
        database.DB_PATH=self.old_path;self.tmp.cleanup()

    async def sql(self,statement,args=()):
        db=await database.get_db()
        try:
            cur=await db.execute(statement,args);rows=await cur.fetchall();await db.commit();return rows
        finally:await db.close()

    async def wallet(self,amount,user=1):
        await self.sql('INSERT INTO tool_credit_balances(user_id,balance) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET balance=excluded.balance',(user,amount))

    async def paid(self,balance=100):
        await self.sql("UPDATE credits SET plan_name='creator',monthly_quota=100,balance=?,sub_status='active',stripe_sub_id='sub_offline' WHERE user_id=1",(balance,))

    async def test_ten_credit_free_gift_creates_one_basic_flipbook(self):
        await self.sql("UPDATE credit_campaigns SET enabled=1,approval_reference='offline-owner-review'")
        await self.sql("INSERT INTO credit_campaign_recipients(campaign_id,user_id,email,plan_name,amount,approved) VALUES(?,1,'flipbook1@example.invalid','free',10,1)",(gifts.CAMPAIGN_ID,))
        with patch.object(gifts,'_now',return_value=gifts.STARTS_AT):await gifts.claim(1)
        result=await tools.consume_flipbook(1,operation(),fingerprint('first'))
        self.assertEqual(result['cost'],10);self.assertEqual(result['funding_source'],'tool')
        self.assertEqual(result['tool_credits'],0);self.assertEqual(result['balance'],3)
        self.assertEqual(result['allowed_modes'],['basic'])

    async def test_free_normal_credits_cannot_replace_a_tool_gift(self):
        await self.sql('UPDATE credits SET balance=100 WHERE user_id=1')
        self.assertFalse((await tools.flipbook_access(1))['can_create'])
        with self.assertRaises(tools.FlipbookCreditError) as failure:await tools.consume_flipbook(1,operation(),fingerprint('not allowed'))
        self.assertEqual(failure.exception.status,402)
        self.assertEqual(await credits.get_balance(1),100)

    async def test_free_partial_wallet_and_daily_allowance_cannot_be_combined(self):
        await self.wallet(7)
        with self.assertRaises(tools.FlipbookCreditError):await tools.consume_flipbook(1,operation(),fingerprint('partial'))
        self.assertEqual(await tools.get_balance(1),7);self.assertEqual(await credits.get_balance(1),3)

    async def test_paid_basic_costs_ten_ordinary_credits(self):
        await self.paid()
        result=await tools.consume_flipbook(1,operation(),fingerprint('paid'))
        self.assertEqual(result['balance'],90);self.assertEqual(result['funding_source'],'ordinary')
        state=(await self.sql('SELECT monthly_quota,stripe_sub_id,sub_status FROM credits WHERE user_id=1'))[0]
        self.assertEqual(tuple(state),(100,'sub_offline','active'))
        delta=(await self.sql("SELECT delta FROM credit_transactions WHERE metadata LIKE '%flipbook%'"))[0][0]
        self.assertEqual(delta,-10)

    async def test_paid_gifts_first_then_ordinary_without_splitting_wallets(self):
        await self.paid();await self.wallet(25)
        for name in ('one','two'):self.assertEqual((await tools.consume_flipbook(1,operation(),fingerprint(name)))['funding_source'],'tool')
        result=await tools.consume_flipbook(1,operation(),fingerprint('three'))
        self.assertEqual(result['funding_source'],'ordinary');self.assertEqual(result['tool_credits'],5);self.assertEqual(result['balance'],90)

    async def test_paid_insufficient_balance_is_not_debited(self):
        await self.paid(9);await self.wallet(9)
        with self.assertRaises(tools.FlipbookCreditError):await tools.consume_flipbook(1,operation(),fingerprint('too few'))
        self.assertEqual(await credits.get_balance(1),9);self.assertEqual(await tools.get_balance(1),9)

    async def test_same_operation_and_artifact_retries_do_not_charge_again(self):
        await self.wallet(10);op=operation();digest=fingerprint('retry')
        first=await tools.consume_flipbook(1,op,digest);second=await tools.consume_flipbook(1,op,digest)
        self.assertFalse(first['already_charged']);self.assertTrue(second['already_charged'])
        self.assertEqual(second['tool_credits'],0);self.assertTrue(second['has_previous']);self.assertFalse(second['can_create'])

    async def test_same_artifact_new_operation_can_be_reopened_free(self):
        await self.paid();first=await tools.consume_flipbook(1,operation(),fingerprint('same document'))
        second=await tools.consume_flipbook(1,operation(),fingerprint('same document'))
        self.assertTrue(second['already_charged']);self.assertEqual(second['operation_id'],first['operation_id']);self.assertEqual(second['balance'],90)

    async def test_operation_reuse_with_different_artifact_is_rejected(self):
        await self.paid();op=operation();await tools.consume_flipbook(1,op,fingerprint('first'))
        with self.assertRaises(tools.FlipbookCreditError) as failure:await tools.consume_flipbook(1,op,fingerprint('changed'))
        self.assertEqual(failure.exception.status,409);self.assertEqual(await credits.get_balance(1),90)

    async def test_invalid_identifiers_never_spend(self):
        await self.wallet(10)
        for op,digest in (('bad',fingerprint('ok')),(operation(),'bad'),(operation(),'A'*64)):
            with self.assertRaises(tools.FlipbookCreditError) as failure:await tools.consume_flipbook(1,op,digest)
            self.assertEqual(failure.exception.status,400)
        self.assertEqual(await tools.get_balance(1),10)

    async def test_concurrent_same_document_is_charged_once(self):
        await self.wallet(10)
        results=await asyncio.gather(*(tools.consume_flipbook(1,operation(),fingerprint('same')) for _ in range(6)))
        self.assertEqual(sum(not r['already_charged'] for r in results),1)
        self.assertEqual(len(await self.sql('SELECT * FROM tool_credit_artifacts')),1)
        self.assertEqual(await tools.get_balance(1),0)

    async def test_concurrent_different_documents_cannot_overspend(self):
        await self.wallet(10)
        results=await asyncio.gather(*(tools.consume_flipbook(1,operation(),fingerprint(str(i))) for i in range(6)),return_exceptions=True)
        self.assertEqual(sum(isinstance(r,dict) for r in results),1)
        self.assertEqual(await tools.get_balance(1),0)

    async def test_receipts_are_scoped_to_the_authenticated_user(self):
        await self.wallet(10,1);await self.wallet(10,2);op=operation();digest=fingerprint('shared PDF')
        await tools.consume_flipbook(1,op,digest)
        result=await tools.consume_flipbook(2,op,digest)
        self.assertFalse(result['already_charged']);self.assertEqual(await tools.get_balance(2),0)
        self.assertEqual(len(await self.sql('SELECT * FROM tool_credit_artifacts')),2)

    async def test_receipt_failure_rolls_back_both_wallet_types(self):
        await self.sql("CREATE TRIGGER fail_artifact BEFORE INSERT ON tool_credit_artifacts BEGIN SELECT RAISE(ABORT,'offline failure'); END")
        await self.wallet(10)
        with self.assertRaises(sqlite3.IntegrityError):await tools.consume_flipbook(1,operation(),fingerprint('gift rollback'))
        self.assertEqual(await tools.get_balance(1),10);self.assertEqual(len(await self.sql('SELECT * FROM tool_credit_transactions')),0)
        await self.wallet(0);await self.paid()
        with self.assertRaises(sqlite3.IntegrityError):await tools.consume_flipbook(1,operation(),fingerprint('ordinary rollback'))
        self.assertEqual(await credits.get_balance(1),100);self.assertEqual(len(await self.sql('SELECT * FROM tool_credit_artifacts')),0)

    async def test_http_authentication_and_server_fixed_cost(self):
        import httpx
        import app as api
        await self.wallet(10)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://offline') as client:
            self.assertEqual((await client.get('/api/credits/tools/flipbook')).status_code,401)
            data={'operation_id':operation(),'artifact_hash':fingerprint('http'),'cost':1,'user_id':2}
            self.assertEqual((await client.post('/api/credits/tools/flipbook',json=data)).status_code,401)
            with patch.object(api,'get_user_or_ip',return_value=(1,'user:1')):
                status=await client.get('/api/credits/tools/flipbook');self.assertEqual(status.headers['cache-control'],'no-store')
                result=await client.post('/api/credits/tools/flipbook',json=data)
                self.assertEqual(result.status_code,200);self.assertEqual(result.json()['cost'],10);self.assertEqual(result.json()['tool_credits'],0)
                invalid=await client.post('/api/credits/tools/flipbook',json={'operation_id':'bad','artifact_hash':'bad'})
                self.assertEqual(invalid.status_code,400)

    async def test_legacy_fifteen_credit_recipient_migration_preserves_approval(self):
        await self.sql('DROP TABLE credit_campaign_recipients')
        old=gifts.SCHEMA_SQL.replace('IN (10,15,25,100,300)','IN (15,25,100,300)')
        db=await database.get_db()
        try:await db.executescript(old)
        finally:await db.close()
        await self.sql("INSERT INTO credit_campaign_recipients(campaign_id,user_id,email,plan_name,amount,approved,claimed_at) VALUES(?,1,'flipbook1@example.invalid','free',15,1,123)",(gifts.CAMPAIGN_ID,))
        before=tuple((await self.sql('SELECT * FROM credit_campaign_recipients WHERE user_id=1'))[0])
        db=await database.get_db()
        try:await gifts.migrate_ten_credit_gifts(db);await gifts.migrate_ten_credit_gifts(db)
        finally:await db.close()
        after=tuple((await self.sql('SELECT * FROM credit_campaign_recipients WHERE user_id=1'))[0]);self.assertEqual(after,before)
        await self.sql("INSERT INTO credit_campaign_recipients(campaign_id,user_id,email,plan_name,amount,approved) VALUES(?,2,'flipbook2@example.invalid','free',10,1)",(gifts.CAMPAIGN_ID,))
        self.assertEqual((await self.sql('PRAGMA integrity_check'))[0][0],'ok')


if __name__=='__main__':unittest.main()
