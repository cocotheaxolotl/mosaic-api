"""Offline regression tests: real SQLite, fake Stripe data, no customer writes."""
import asyncio
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import database
import annual_credits as annual
import credits


def ts(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


class CalendarTests(unittest.TestCase):
    def test_month_end_does_not_drift(self):
        anchor = ts("2026-01-31T12:13:14")
        self.assertEqual(annual.month_after(anchor, 1), ts("2026-02-28T12:13:14"))
        self.assertEqual(annual.month_after(anchor, 2), ts("2026-03-31T12:13:14"))

    def test_leap_year(self):
        self.assertEqual(annual.month_after(ts("2028-01-31T01:02:03"), 1), ts("2028-02-29T01:02:03"))
        self.assertEqual(annual.month_after(ts("2028-02-29T01:02:03"), 12), ts("2029-02-28T01:02:03"))


class RenewalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="annual-credit-test-")
        self.old_path = database.DB_PATH
        database.DB_PATH = str(Path(self.tmp.name)/"test.db")
        await database.init_db()
        self.start = ts("2026-10-16T16:01:15")
        self.now = ts("2026-11-19T12:00:00")
        self.sub = {"id":"sub_test", "customer":"cus_test", "status":"active",
                    "items":{"data":[{"current_period_start":self.start,
                    "current_period_end":annual.month_after(self.start,12),
                    "price":{"recurring":{"interval":"year","interval_count":1}}}]},
                    "latest_invoice":{"status":"paid"}}
        db = await database.get_db()
        await db.execute("INSERT INTO users(id,email,password_hash,stripe_customer_id) VALUES(1,'test@example.invalid','unused','cus_test')")
        await db.execute("INSERT INTO credits(user_id,balance,monthly_quota,plan_name,stripe_sub_id,sub_status) VALUES(1,0,100,'creator','sub_test','active')")
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata,created_at) VALUES(1,100,'subscription_start',?,?)",
                         (json.dumps({"stripe_sub_id":"sub_test"}),self.start+4))
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,created_at) VALUES(1,-100,'generation',?)", (self.start+10,))
        await db.execute("UPDATE annual_credit_policy SET live_from=?", (ts("2026-11-01T00:00:00"),))
        await db.commit(); await db.close()

    async def asyncTearDown(self):
        database.DB_PATH = self.old_path
        self.tmp.cleanup()

    async def sql(self, statement, args=()):
        db=await database.get_db()
        try:
            cursor=await db.execute(statement,args)
            rows=await cursor.fetchall()
            await db.commit()
            return rows
        finally:
            await db.close()

    async def balance(self):
        return (await self.sql("SELECT balance FROM credits WHERE user_id=1"))[0][0]

    async def process(self):
        return await annual.process_subscription(1,self.sub,self.now)

    async def test_paid_annual_refills_without_a_new_invoice(self):
        await self.process()
        self.assertEqual(await self.balance(),100)
        rows=await self.sql("SELECT delta,metadata FROM credit_transactions WHERE reason='subscription_renewal'")
        self.assertEqual(len(rows),1)
        self.assertEqual(json.loads(rows[0][1])["period_start"],annual.month_after(self.start,1))

    async def test_retry_and_concurrent_workers_never_duplicate(self):
        await asyncio.gather(*(self.process() for _ in range(6)))
        self.assertEqual(await self.balance(),100)
        self.assertEqual((await self.sql("SELECT COUNT(*) FROM credit_transactions WHERE reason='subscription_renewal'"))[0][0],1)
        await self.sql("UPDATE credits SET balance=97 WHERE user_id=1")
        await self.process()
        self.assertEqual(await self.balance(),97)

    async def test_historical_backfill_requires_review_and_stays_blocked(self):
        await self.sql("UPDATE annual_credit_policy SET live_from=?",(self.now,))
        await self.process(); await self.process()
        self.assertEqual(await self.balance(),0)
        self.assertEqual((await self.sql("SELECT status FROM annual_credit_periods WHERE period_start=?",(annual.month_after(self.start,1),)))[0][0],"pending_review")
        self.now=ts("2026-12-19T12:00:00")
        await self.process()
        self.assertEqual(await self.balance(),100)

    async def test_cutoff_survives_restart(self):
        before=(await self.sql("SELECT live_from FROM annual_credit_policy"))[0][0]
        await database.init_db()
        self.assertEqual((await self.sql("SELECT live_from FROM annual_credit_policy"))[0][0],before)

    async def test_not_before_anniversary(self):
        self.now=annual.month_after(self.start,1)-1
        await self.process()
        self.assertEqual(await self.balance(),0)
        self.now+=1
        await self.process()
        self.assertEqual(await self.balance(),100)

    async def test_creator_does_not_accumulate_expired_months(self):
        self.now=ts("2027-01-19T12:00:00")
        await self.process()
        self.assertEqual(await self.balance(),100)
        statuses=[row[0] for row in await self.sql("SELECT status FROM annual_credit_periods ORDER BY period_start")]
        self.assertEqual(statuses,["already_granted","expired","expired","granted"])

    async def test_studio_preserves_and_catches_up_rollover(self):
        await self.sql("UPDATE credits SET plan_name='studio',monthly_quota=1500,balance=40")
        self.now=ts("2027-01-19T12:00:00")
        await self.process(); await self.process()
        self.assertEqual(await self.balance(),4540)

    async def test_packs_are_preserved(self):
        await credits.add_credits(1,100,"credit_pack_purchase")
        await self.process()
        self.assertEqual(await self.balance(),200)
        # Real renewal deltas may differ from the quota. Replay must use quota.
        ok,_=await credits.consume_credits(1,80,"generation")
        self.assertTrue(ok)
        self.now=ts("2026-12-19T12:00:00")
        await self.process()
        self.assertEqual(await self.balance(),200)

    async def test_concurrent_purchase_and_renewal_preserve_both(self):
        await asyncio.gather(self.process(),credits.add_credits(1,100,"credit_pack_purchase"))
        self.assertEqual(await self.balance(),200)

    async def test_concurrent_consumption_cannot_overwrite_refill(self):
        results=await asyncio.gather(self.process(),*(credits.consume_credits(1,1,"generation") for _ in range(5)))
        successes=sum(int(result[0]) for result in results[1:])
        self.assertEqual(await self.balance(),100-successes)

    async def test_cancel_at_period_end_keeps_paid_access(self):
        self.sub["cancel_at_period_end"]=True
        await self.process()
        self.assertEqual(await self.balance(),100)

    async def test_ineligible_subscriptions_never_change_balance(self):
        for field,value in (("status","canceled"),("status","past_due"),("customer","cus_other"),("id","sub_other")):
            with self.subTest(field=field,value=value):
                old=self.sub[field];self.sub[field]=value
                await self.process();self.assertEqual(await self.balance(),0)
                self.sub[field]=old

    async def test_unpaid_annual_term_never_refills(self):
        self.sub["latest_invoice"]["status"]="open"
        await self.process();self.assertEqual(await self.balance(),0)

    async def test_monthly_subscription_untouched(self):
        self.sub["items"]["data"][0]["price"]["recurring"]["interval"]="month"
        await self.process();self.assertEqual(await self.balance(),0)

    async def test_expired_term_never_refills(self):
        self.now=annual.month_after(self.start,12)
        await self.process();self.assertEqual(await self.balance(),0)

    async def test_local_free_or_canceled_account_never_refills(self):
        for plan,status in (("free","active"),("creator","canceled"),("creator","trial")):
            await self.sql("UPDATE credits SET plan_name=?,sub_status=?",(plan,status))
            await self.process();self.assertEqual(await self.balance(),0)

    async def test_legacy_top_level_stripe_periods(self):
        item=self.sub["items"]["data"][0]
        self.sub["current_period_start"]=item.pop("current_period_start")
        self.sub["current_period_end"]=item.pop("current_period_end")
        await self.process();self.assertEqual(await self.balance(),100)

    async def test_new_paid_year_and_its_retries_grant_once(self):
        item=self.sub["items"]["data"][0]
        item["current_period_start"]=annual.month_after(self.start,12)
        item["current_period_end"]=annual.month_after(self.start,24)
        self.now=item["current_period_start"]+1
        await self.process(); await self.process()
        self.assertEqual(await self.balance(),100)

    async def test_monthly_invoice_keeps_existing_renewal(self):
        from unittest.mock import patch, AsyncMock
        import stripe_integration as billing
        monthly=copy.deepcopy(self.sub)
        monthly["items"]["data"][0]["price"]["recurring"]["interval"]="month"
        event={"type":"invoice.paid","data":{"object":{"id":"in_test","customer":"cus_test","subscription":"sub_test","billing_reason":"subscription_cycle","amount_paid":999}}}
        with patch.object(billing.stripe.Webhook,"construct_event",return_value=event), patch.object(billing.stripe.Subscription,"retrieve",return_value=monthly), patch.object(billing,"_record_affiliate_payout",new_callable=AsyncMock):
            await billing.handle_webhook(b"unused","unused")
        self.assertEqual(await self.balance(),100)

    async def test_overlapping_old_monthly_invoice_cannot_reset_new_plan(self):
        from unittest.mock import patch, AsyncMock
        import stripe_integration as billing
        monthly=copy.deepcopy(self.sub)
        monthly["id"]="sub_old"
        monthly["items"]["data"][0]["price"]["recurring"]["interval"]="month"
        event={"type":"invoice.paid","data":{"object":{"id":"in_test","customer":"cus_test","subscription":"sub_old","billing_reason":"subscription_cycle","amount_paid":999}}}
        with patch.object(billing.stripe.Webhook,"construct_event",return_value=event), patch.object(billing.stripe.Subscription,"retrieve",return_value=monthly), patch.object(billing,"_record_affiliate_payout",new_callable=AsyncMock):
            await billing.handle_webhook(b"unused","unused")
        self.assertEqual(await self.balance(),0)

    async def test_annual_invoice_new_api_and_worker_share_one_ledger(self):
        from unittest.mock import patch, AsyncMock
        import stripe_integration as billing
        event={"type":"invoice.paid","data":{"object":{"id":"in_test","customer":"cus_test","parent":{"subscription_details":{"subscription":"sub_test"}},"billing_reason":"subscription_cycle","amount_paid":9590}}}
        with patch.object(billing.stripe.Webhook,"construct_event",return_value=event), patch.object(billing.stripe.Subscription,"retrieve",return_value=self.sub), patch.object(billing,"_record_affiliate_payout",new_callable=AsyncMock), patch.object(annual.time,"time",return_value=self.now):
            await billing.handle_webhook(b"unused","unused")
            await billing.handle_webhook(b"unused","unused")
        await self.process()
        self.assertEqual(await self.balance(),100)
        self.assertEqual((await self.sql("SELECT COUNT(*) FROM credit_transactions WHERE reason='subscription_renewal'"))[0][0],1)

    async def test_annual_invoice_cannot_bypass_historical_review(self):
        from unittest.mock import patch, AsyncMock
        import stripe_integration as billing
        await self.sql("UPDATE annual_credit_policy SET live_from=?",(self.now,))
        event={"type":"invoice.paid","data":{"object":{"id":"in_test","customer":"cus_test","subscription":"sub_test","billing_reason":"subscription_cycle","amount_paid":9590}}}
        with patch.object(billing.stripe.Webhook,"construct_event",return_value=event), patch.object(billing.stripe.Subscription,"retrieve",return_value=self.sub), patch.object(billing,"_record_affiliate_payout",new_callable=AsyncMock), patch.object(annual.time,"time",return_value=self.now):
            await billing.handle_webhook(b"unused","unused")
        self.assertEqual(await self.balance(),0)


if __name__=="__main__":
    unittest.main()
