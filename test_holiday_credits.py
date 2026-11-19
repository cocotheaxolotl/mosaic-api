"""Isolated non-API gift wallet: real SQLite, mocked paid providers."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
import database
import credits
import tool_credits as tools
import holiday_credits as gifts


class HolidayCreditsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="holiday-tools-")
        self.old_path=database.DB_PATH;database.DB_PATH=str(Path(self.tmp.name)/"test.db")
        await database.init_db()
        await self.sql("INSERT INTO users(id,email,password_hash) VALUES(1,'gift@example.invalid','unused')")
        await self.sql("INSERT INTO credits(user_id,balance,monthly_quota,plan_name,sub_status) VALUES(1,3,3,'free','none')")
        await self.sql("INSERT INTO credit_transactions(user_id,delta,reason) VALUES(1,3,'signup_bonus')")
        self.clock=patch.object(gifts,"_now",return_value=gifts.STARTS_AT);self.clock.start()

    async def asyncTearDown(self):
        self.clock.stop();database.DB_PATH=self.old_path;self.tmp.cleanup()

    async def sql(self,sql,args=()):
        db=await database.get_db()
        try:
            cur=await db.execute(sql,args);rows=await cur.fetchall();await db.commit();return rows
        finally:await db.close()

    async def authorize(self,amount=15):
        await self.sql("UPDATE credit_campaigns SET enabled=1,approval_reference='offline-owner-approval'")
        await self.sql("INSERT INTO credit_campaign_recipients(campaign_id,user_id,email,plan_name,amount,approved) VALUES(?,1,'gift@example.invalid','free',?,1)",(gifts.CAMPAIGN_ID,amount))

    async def test_disabled_by_default_and_no_automatic_enrollment(self):
        self.assertFalse((await gifts.get_status(1))["active"])
        self.assertEqual(len(await self.sql("SELECT * FROM credit_campaign_recipients")),0)
        with self.assertRaises(gifts.GiftUnavailable):await gifts.claim(1)
        self.assertEqual(await tools.get_balance(1),0)

    async def test_campaign_and_recipient_approval_both_required(self):
        await self.authorize()
        for sql in ("UPDATE credit_campaigns SET approval_reference=NULL","UPDATE credit_campaign_recipients SET approved=0"):
            await self.sql(sql)
            with self.assertRaises(gifts.GiftUnavailable):await gifts.claim(1)
            await self.sql("UPDATE credit_campaigns SET approval_reference='offline-owner-approval'")
            await self.sql("UPDATE credit_campaign_recipients SET approved=1")
        self.assertEqual(await tools.get_balance(1),0)

    async def test_other_account_and_changed_email_cannot_claim(self):
        await self.authorize()
        with self.assertRaises(gifts.GiftUnavailable):await gifts.claim(2)
        await self.sql("UPDATE users SET email='changed@example.invalid' WHERE id=1")
        with self.assertRaises(gifts.GiftUnavailable):await gifts.claim(1)

    async def test_paris_redemption_boundaries(self):
        await self.authorize()
        for instant,active in ((gifts.STARTS_AT-1,False),(gifts.STARTS_AT,True),(gifts.ENDS_AT-1,True),(gifts.ENDS_AT,False)):
            with patch.object(gifts,"_now",return_value=instant):
                self.assertEqual((await gifts.get_status(1))["active"],active)
                if not active:
                    with self.assertRaises(gifts.GiftUnavailable):await gifts.claim(1)

    async def test_public_status_exposes_no_recipient_information(self):
        await self.authorize();status=await gifts.get_status()
        self.assertEqual(status["credit_type"],"non_api")
        self.assertNotIn("amount",status);self.assertNotIn("eligible",status)

    async def test_claim_adds_only_tool_credits(self):
        await self.authorize();result=await gifts.claim(1)
        self.assertEqual(result["tool_credits"],15)
        self.assertEqual(result["credit_type"],"non_api")
        self.assertEqual(await credits.get_balance(1),3)
        self.assertEqual(await tools.get_balance(1),15)
        self.assertEqual(len(await self.sql("SELECT * FROM credit_transactions")),1)

    async def test_concurrent_claims_and_retries_grant_once(self):
        await self.authorize();results=await asyncio.gather(*(gifts.claim(1) for _ in range(6)))
        self.assertEqual(sum(not r["already_claimed"] for r in results),1)
        await credits.consume_credits(1,3,"generation")
        await tools.consume_download(1,"/fr/maze/");await tools.consume_download(1,"/sudoku/")
        self.assertTrue((await gifts.claim(1))["already_claimed"])
        self.assertEqual(await tools.get_balance(1),13)
        self.assertEqual(await credits.get_balance(1),0)
        self.assertEqual(len(await self.sql("SELECT * FROM tool_credit_transactions WHERE reason='christmas_non_api_gift'")),1)

    async def test_free_allowance_is_used_before_lasting_tool_gift(self):
        await self.authorize();await gifts.claim(1)
        for _ in range(3):self.assertFalse((await tools.consume_download(1,"/fr/maze/"))["used_tool_credit"])
        self.assertEqual(await tools.get_balance(1),15)
        result=await tools.consume_download(1,"/fr/maze/")
        self.assertTrue(result["used_tool_credit"]);self.assertEqual(result["tool_credits"],14)
        await self.sql("UPDATE credits SET updated_at=0")
        self.assertEqual(await credits.reset_free_daily_quota_if_needed(1),3)
        self.assertEqual(await tools.get_balance(1),14)

    async def test_paid_customer_spends_tool_gift_before_api_capable_balance(self):
        await self.sql("UPDATE credits SET plan_name='creator',monthly_quota=100,balance=100")
        await self.authorize(25);await gifts.claim(1)
        result=await tools.consume_download(1,"/wordsearch/")
        self.assertTrue(result["used_tool_credit"])
        self.assertEqual(await credits.get_balance(1),100)
        self.assertEqual(await tools.get_balance(1),24)

    async def test_ordinary_credit_consumption_cannot_spend_tool_gift(self):
        await self.authorize();await gifts.claim(1)
        await credits.consume_credits(1,3,"generation")
        self.assertEqual(await credits.consume_credits(1,1,"generation",{"feature":"ai-image"}),(False,0))
        self.assertEqual(await tools.get_balance(1),15)

    async def test_unknown_or_ai_route_cannot_use_tool_wallet(self):
        await self.authorize();await gifts.claim(1)
        await credits.consume_credits(1,3,"generation")
        for path in ("/ai-image/","/coloring/","/mosaic/","/api/ai-image","https://evil.invalid/maze/","/fr/maze/extra/"):
            self.assertFalse((await tools.consume_download(1,path))["ok"])
        self.assertEqual(await tools.get_balance(1),15)

    async def test_browser_tools_locales_are_allowed(self):
        self.assertEqual(tools.browser_tool('/fr/maze/'),'maze')
        self.assertEqual(tools.browser_tool('/ja/sudoku/'),'sudoku')
        self.assertEqual(tools.browser_tool('/crossword/'),'crossword')
        self.assertIsNone(tools.browser_tool('/ai-image/'))

    async def test_unknown_tool_preserves_legacy_ordinary_consumption(self):
        result=await tools.consume_download(1,"/other-tool/")
        self.assertTrue(result["ok"]);self.assertFalse(result["used_tool_credit"])
        self.assertEqual(await credits.get_balance(1),2)

    async def test_concurrent_exports_never_overspend_tool_balance(self):
        await self.authorize();await gifts.claim(1)
        await credits.consume_credits(1,3,"generation")
        results=await asyncio.gather(*(tools.consume_download(1,"/maze/") for _ in range(20)))
        self.assertEqual(sum(r["ok"] for r in results),15)
        self.assertEqual(await tools.get_balance(1),0)
        self.assertEqual(await credits.get_balance(1),0)

    async def test_renewal_upgrade_and_trial_expiry_leave_tool_wallet_unchanged(self):
        await self.authorize(300);await gifts.claim(1)
        await credits.set_plan(1,"creator",100,"sub_test")
        self.assertEqual(await credits.get_balance(1),100)
        await credits.renew_credits(1)
        self.assertEqual(await credits.get_balance(1),100)
        await self.sql("UPDATE credits SET sub_status='trial'")
        await self.sql("INSERT INTO trial_codes(code,created_for,used_by,expires_at) VALUES('TEST','gift@example.invalid',1,1)")
        await credits.expire_trial_if_needed(1)
        self.assertEqual(await credits.get_balance(1),3)
        self.assertEqual(await tools.get_balance(1),300)

    async def test_failed_claim_rolls_back_wallet_and_receipt(self):
        await self.authorize()
        with patch.object(credits,"_record_operation",side_effect=RuntimeError('offline failure')):
            with self.assertRaises(RuntimeError):await gifts.claim(1)
        self.assertEqual(await tools.get_balance(1),0)
        self.assertIsNone((await self.sql("SELECT claimed_at FROM credit_campaign_recipients"))[0][0])
        self.assertEqual(len(await self.sql("SELECT * FROM tool_credit_transactions")),0)

    async def test_profile_reports_separate_balances(self):
        import auth
        await self.authorize();await gifts.claim(1)
        profile=await auth.get_me(1);info=await credits.get_plan_info(1)
        self.assertEqual(profile['credits'],3);self.assertEqual(profile['tool_credits'],15)
        self.assertEqual(info['balance'],3);self.assertEqual(info['tool_credits'],15)

    async def test_http_auth_and_api_spending_are_isolated(self):
        import httpx
        import app as api
        await self.authorize();await gifts.claim(1)
        await self.sql("UPDATE credits SET balance=0,plan_name='creator',monthly_quota=100")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://offline-test') as client:
            self.assertEqual((await client.post('/api/credits/tools/consume',json={'tool':'/maze/'})).status_code,401)
            with patch.object(api,'get_user_or_ip',return_value=(1,'user:1')):
                response=await client.post('/api/credits/consume',json={'feature':'ai-image','variant':'high'})
                self.assertEqual(response.status_code,402)
                response=await client.post('/api/credits/tools/consume',json={'tool':'/fr/maze/','amount':999999})
                self.assertEqual(response.status_code,200);self.assertEqual(response.json()['tool_credits'],14)
                with patch.object(api,'_ai_text_to_image',new_callable=AsyncMock) as provider:
                    response=await client.post('/api/ai-image',data={'prompt':'offline test','quality':'low'})
                    self.assertEqual(response.status_code,402);provider.assert_not_called()
        self.assertEqual(await credits.get_balance(1),0)

    async def test_free_image_request_never_calls_paid_provider(self):
        import httpx
        import app as api
        await self.authorize();await gifts.claim(1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://offline-test') as client:
            with patch.object(api,'get_user_or_ip',return_value=(1,'user:1')),patch.object(api,'_ai_text_to_image',new_callable=AsyncMock) as provider:
                response=await client.post('/api/ai-image',data={'prompt':'offline test'})
                self.assertEqual(response.status_code,403);provider.assert_not_called()

    async def test_image_quality_cost_uses_ordinary_credits(self):
        import httpx
        import app as api
        await self.authorize(100);await gifts.claim(1)
        await self.sql("UPDATE credits SET plan_name='pro',monthly_quota=400,balance=12")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://offline-test') as client:
            with patch.object(api,'get_user_or_ip',return_value=(1,'user:1')):
                response=await client.post('/api/credits/consume',json={'feature':'ai-image','variant':'high'})
                self.assertEqual(response.status_code,200);self.assertEqual(response.json()['remaining'],2)
        self.assertEqual(await tools.get_balance(1),100)


if __name__=='__main__':unittest.main()
