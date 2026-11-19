"""Real SQLite gift lifecycle tests; no Stripe calls, emails or real recipients."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import database
import credits
import holiday_credits as gifts


class HolidayCreditsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="holiday-credits-")
        self.old_path = database.DB_PATH
        database.DB_PATH = str(Path(self.tmp.name)/"test.db")
        await database.init_db()
        await self.sql("INSERT INTO users(id,email,password_hash) VALUES(1,'gift@example.invalid','unused')")
        await self.sql("INSERT INTO credits(user_id,balance,monthly_quota,plan_name,sub_status) VALUES(1,3,3,'free','none')")
        await self.sql("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(1,3,'signup_bonus','{}')")
        self.clock = patch.object(gifts,"_now",return_value=gifts.STARTS_AT)
        self.clock.start()

    async def asyncTearDown(self):
        self.clock.stop()
        database.DB_PATH = self.old_path
        self.tmp.cleanup()

    async def sql(self,sql,args=()):
        db = await database.get_db()
        try:
            cur = await db.execute(sql,args)
            result = await cur.fetchall()
            await db.commit()
            return result
        finally:
            await db.close()

    async def authorize(self,amount=15):
        await self.sql("UPDATE credit_campaigns SET enabled=1,approval_reference='offline-test-owner-approval'")
        await self.sql("INSERT INTO credit_campaign_recipients(campaign_id,user_id,email,plan_name,amount,approved) VALUES(?,1,'gift@example.invalid','free',?,1)",(gifts.CAMPAIGN_ID,amount))

    async def balance(self):
        return (await self.sql("SELECT balance FROM credits WHERE user_id=1"))[0][0]

    async def test_disabled_by_default_no_automatic_recipients(self):
        self.assertFalse((await gifts.get_status(1))["active"])
        self.assertEqual(len(await self.sql("SELECT * FROM credit_campaign_recipients")),0)
        with self.assertRaises(gifts.GiftUnavailable): await gifts.claim(1)
        self.assertEqual(await self.balance(),3)

    async def test_campaign_enabled_without_approval_is_blocked(self):
        await self.sql("UPDATE credit_campaigns SET enabled=1")
        self.assertFalse((await gifts.get_status())["active"])

    async def test_recipient_needs_individual_approval(self):
        await self.authorize()
        await self.sql("UPDATE credit_campaign_recipients SET approved=0")
        with self.assertRaises(gifts.GiftUnavailable): await gifts.claim(1)
        self.assertEqual(await self.balance(),3)

    async def test_other_account_cannot_claim_for_someone_else(self):
        await self.authorize()
        with self.assertRaises(gifts.GiftUnavailable): await gifts.claim(2)
        self.assertEqual(await self.balance(),3)

    async def test_reassigned_email_requires_fresh_review(self):
        await self.authorize()
        await self.sql("UPDATE users SET email='changed@example.invalid' WHERE id=1")
        with self.assertRaises(gifts.GiftUnavailable): await gifts.claim(1)

    async def test_public_status_contains_no_recipient_information(self):
        await self.authorize()
        status = await gifts.get_status()
        self.assertTrue(status["active"])
        self.assertNotIn("eligible",status)
        self.assertNotIn("amount",status)

    async def test_paris_date_boundaries(self):
        await self.authorize()
        for instant, active in ((gifts.STARTS_AT-1,False),(gifts.STARTS_AT,True),(gifts.ENDS_AT-1,True),(gifts.ENDS_AT,False)):
            with patch.object(gifts,"_now",return_value=instant):
                self.assertEqual((await gifts.get_status(1))["active"],active)
                if not active:
                    with self.assertRaises(gifts.GiftUnavailable): await gifts.claim(1)
        self.assertEqual(await self.balance(),3)

    async def test_retry_and_concurrency_add_only_one_gift(self):
        await self.authorize()
        results = await asyncio.gather(*(gifts.claim(1) for _ in range(6)))
        self.assertEqual(sum(not r["already_claimed"] for r in results),1)
        self.assertEqual(await self.balance(),18)
        self.assertEqual(len(await self.sql("SELECT * FROM credit_transactions WHERE reason='promotional_gift'")),1)
        await credits.consume_credits(1,5,"generation")
        self.assertTrue((await gifts.claim(1))["already_claimed"])
        self.assertEqual(await self.balance(),13)

    async def test_client_cannot_choose_a_larger_amount(self):
        await self.authorize(15)
        result = await gifts.claim(1)
        self.assertEqual(result["amount"],15)
        self.assertEqual((await gifts.get_status(1))["amount"],15)

    async def test_free_daily_reset_preserves_gift_and_spends_daily_first(self):
        await self.authorize()
        await gifts.claim(1)
        await credits.consume_credits(1,2,"generation")
        await self.sql("UPDATE credits SET updated_at=0")
        self.assertEqual(await credits.reset_free_daily_quota_if_needed(1),18)
        await credits.consume_credits(1,10,"generation")
        await self.sql("UPDATE credits SET updated_at=0")
        self.assertEqual(await credits.reset_free_daily_quota_if_needed(1),11)
        ok, balance = await credits.consume_credits(1,12,"generation")
        self.assertFalse(ok); self.assertEqual(balance,11)

    async def test_no_free_gift_means_existing_daily_allowance(self):
        await credits.consume_credits(1,3,"generation")
        await self.sql("UPDATE credits SET updated_at=0")
        self.assertEqual(await credits.reset_free_daily_quota_if_needed(1),3)

    async def test_monthly_renewal_and_upgrade_keep_unspent_gift(self):
        await self.authorize(25)
        await gifts.claim(1)
        await credits.set_plan(1,"creator",100,"sub_test")
        self.assertEqual(await self.balance(),125)
        await credits.consume_credits(1,110,"generation")
        await credits.renew_credits(1)
        self.assertEqual(await self.balance(),115)
        await credits.set_plan(1,"pro",400,"sub_pro")
        self.assertEqual(await self.balance(),415)

    async def test_studio_rollover_keeps_gift_once(self):
        await self.sql("UPDATE credits SET plan_name='studio',monthly_quota=1200,balance=0")
        await self.sql("INSERT INTO credit_transactions(user_id,delta,reason) VALUES(1,-3,'generation')")
        await self.authorize(300)
        await gifts.claim(1)
        await credits.renew_credits(1,operation_key="in_studio")
        await credits.renew_credits(1,operation_key="in_studio")
        self.assertEqual(await self.balance(),1500)

    async def test_trial_expiry_keeps_gift_and_only_standard_free_rights(self):
        await self.sql("UPDATE credits SET plan_name='studio',monthly_quota=1200,balance=0,sub_status='trial'")
        await self.sql("INSERT INTO credit_transactions(user_id,delta,reason) VALUES(1,-3,'generation')")
        await self.sql("INSERT INTO trial_codes(code,created_for,used_by,expires_at) VALUES('TEST','gift@example.invalid',1,1)")
        await self.authorize(300)
        await gifts.claim(1)
        await credits.expire_trial_if_needed(1)
        self.assertEqual(await self.balance(),303)
        self.assertEqual(tuple((await self.sql("SELECT plan_name,sub_status,monthly_quota FROM credits WHERE user_id=1"))[0]),("free","none",3))
        await credits.consume_credits(1,8,"generation")
        await self.sql("UPDATE credits SET updated_at=0")
        self.assertEqual(await credits.reset_free_daily_quota_if_needed(1),298)

    async def test_gift_transaction_rolls_back_with_receipt_failure(self):
        await self.authorize()
        with patch.object(credits,"_record_operation",side_effect=RuntimeError("offline failure")):
            with self.assertRaises(RuntimeError): await gifts.claim(1)
        self.assertEqual(await self.balance(),3)
        self.assertIsNone((await self.sql("SELECT claimed_at FROM credit_campaign_recipients"))[0][0])
        self.assertEqual(len(await self.sql("SELECT * FROM credit_transactions WHERE reason='promotional_gift'")),0)

    async def test_disabled_campaign_cannot_be_claimed_after_previous_success(self):
        await self.authorize(); await gifts.claim(1)
        await self.sql("UPDATE credit_campaigns SET enabled=0")
        with self.assertRaises(gifts.GiftUnavailable): await gifts.claim(1)
        self.assertEqual(await self.balance(),18)


    async def test_http_auth_and_disabled_campaign(self):
        import httpx
        import app as api
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url="http://offline-test") as client:
            response = await client.get("/api/promotions/christmas-2026")
            self.assertEqual(response.status_code,200)
            self.assertFalse(response.json()["active"])
            self.assertEqual(response.headers["cache-control"],"no-store")
            self.assertEqual((await client.get("/api/promotions/christmas-2026",headers={"Authorization":"Bearer expired"})).status_code,401)
            self.assertEqual((await client.post("/api/promotions/christmas-2026/claim")).status_code,401)
            with patch.object(api,"get_user_or_ip",return_value=(1,"user:1")):
                response = await client.post("/api/promotions/christmas-2026/claim",json={"amount":999999,"user_id":2})
                self.assertEqual(response.status_code,403)
        self.assertEqual(await self.balance(),3)


if __name__ == "__main__":
    unittest.main()
