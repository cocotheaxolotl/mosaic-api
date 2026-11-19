"""Payment lifecycle regression tests. Temporary SQLite, no real Stripe writes."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
import database
import credits
import stripe_integration as billing


class BillingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="billing-lifecycle-")
        self.old_path=database.DB_PATH;database.DB_PATH=str(Path(self.tmp.name)/"test.db")
        await database.init_db()
        db=await database.get_db()
        await db.execute("INSERT INTO users(id,email,password_hash,stripe_customer_id) VALUES(1,'test@example.invalid','unused','cus_test')")
        await db.execute("INSERT INTO credits(user_id,balance,monthly_quota,plan_name,stripe_sub_id,sub_status) VALUES(1,0,100,'creator','sub_current','active')")
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason,metadata) VALUES(1,100,'subscription_start',?)",(json.dumps({"stripe_sub_id":"sub_current","quota":100}),))
        await db.execute("INSERT INTO credit_transactions(user_id,delta,reason) VALUES(1,-100,'generation')")
        await db.commit();await db.close()
        self.sub={"id":"sub_current","customer":"cus_test","status":"active","current_period_start":1,"current_period_end":4000000000,
                  "items":{"data":[{"price":{"recurring":{"interval":"month"}}}]}}

    async def asyncTearDown(self):
        database.DB_PATH=self.old_path;self.tmp.cleanup()

    async def sql(self,sql,args=()):
        db=await database.get_db()
        try:
            cur=await db.execute(sql,args);rows=await cur.fetchall();await db.commit();return rows
        finally:await db.close()

    async def balance(self):return (await self.sql("SELECT balance FROM credits WHERE user_id=1"))[0][0]

    async def event(self,kind,data,subscription=None):
        event={"type":kind,"data":{"object":data}}
        with patch.object(billing.stripe.Webhook,"construct_event",return_value=event), patch.object(billing.stripe.Subscription,"retrieve",return_value=subscription or {**self.sub,"id":data.get("subscription",self.sub["id"])}), patch.object(billing,"_record_affiliate_payout",new_callable=AsyncMock):
            return await billing.handle_webhook(b"test","test")

    def invoice(self,id="in_month_1"):
        return {"id":id,"customer":"cus_test","subscription":"sub_current","billing_reason":"subscription_cycle","amount_paid":999}

    def checkout(self,pack=None):
        metadata={"coco_user_id":"1"}
        metadata.update({"pack":pack} if pack else {"plan":"creator"})
        return {"id":"cs_pack" if pack else "cs_new_plan","customer":"cus_test","subscription":"sub_new","payment_status":"paid","metadata":metadata}

    async def test_paid_monthly_invoice_is_not_reapplied_after_consumption(self):
        data=self.invoice();await self.event("invoice.paid",data)
        await credits.consume_credits(1,3,"generation")
        await self.event("invoice.paid",data)
        self.assertEqual(await self.balance(),97)

    async def test_legacy_monthly_invoice_retry_does_not_refill_consumed_credits(self):
        await credits.renew_credits(1)
        await credits.consume_credits(1,7,"generation")
        await self.event("invoice.paid",self.invoice())
        self.assertEqual(await self.balance(),93)

    async def test_monthly_renewal_preserves_pack_and_gift(self):
        await credits.add_credits(1,20,"credit_pack_purchase")
        await credits.add_credits(1,100,"customer_service_gift")
        await self.event("invoice.paid",self.invoice())
        self.assertEqual(await self.balance(),220)

    async def test_studio_monthly_renewal_preserves_rollover_and_is_idempotent(self):
        await self.sql("UPDATE credits SET plan_name='studio',monthly_quota=1500,balance=1450")
        await self.event("invoice.paid",self.invoice());await self.event("invoice.paid",self.invoice())
        self.assertEqual(await self.balance(),2950)

    async def test_completed_and_async_success_do_not_double_a_pack(self):
        data=self.checkout("pack_20")
        await self.event("checkout.session.completed",data)
        await self.event("checkout.session.async_payment_succeeded",data)
        self.assertEqual(await self.balance(),20)
        row=(await self.sql("SELECT metadata FROM credit_transactions WHERE reason='credit_pack_purchase'"))[0]
        self.assertIsInstance(json.loads(row[0]),dict)

    async def test_concurrent_pack_deliveries_are_atomic(self):
        data=self.checkout("pack_20")
        await asyncio.gather(*(billing._fulfill_checkout_session(data) for _ in range(6)))
        self.assertEqual(await self.balance(),20)

    async def test_unpaid_completed_session_waits_for_async_success(self):
        data=self.checkout("pack_20");data["payment_status"]="unpaid"
        await self.event("checkout.session.completed",data)
        self.assertEqual(await self.balance(),0)
        data["payment_status"]="paid"
        await self.event("checkout.session.async_payment_succeeded",data)
        self.assertEqual(await self.balance(),20)

    async def test_failed_async_payment_cannot_grant_credits(self):
        await self.event("checkout.session.async_payment_failed",self.checkout("pack_20"))
        self.assertEqual(await self.balance(),0)

    async def test_paid_activation_retains_gift_and_retry_does_not_reset_usage(self):
        await credits.add_credits(1,100,"customer_service_gift")
        data=self.checkout();await self.event("checkout.session.completed",data)
        self.assertEqual(await self.balance(),200)
        await credits.consume_credits(1,8,"generation")
        await self.event("checkout.session.async_payment_succeeded",data)
        self.assertEqual(await self.balance(),192)

    async def test_old_subscription_checkout_retry_cannot_reset_current_balance(self):
        data=self.checkout();data["subscription"]="sub_current"
        await credits.add_credits(1,100,"customer_service_gift")
        await self.event("checkout.session.completed",data)
        self.assertEqual(await self.balance(),100)

    async def test_old_contract_state_events_do_not_disable_current_contract(self):
        for kind in ("customer.subscription.updated","customer.subscription.deleted"):
            await self.event(kind,{"id":"sub_old","customer":"cus_test","status":"canceled"})
            self.assertEqual((await self.sql("SELECT sub_status FROM credits"))[0][0],"active")

    async def test_old_failed_invoice_new_api_does_not_disable_current_contract(self):
        data={"customer":"cus_test","parent":{"subscription_details":{"subscription":"sub_old"}}}
        await self.event("invoice.payment_failed",data)
        self.assertEqual((await self.sql("SELECT sub_status FROM credits"))[0][0],"active")

    async def test_late_failure_uses_canonical_paid_active_status(self):
        await self.event("invoice.payment_failed",self.invoice())
        self.assertEqual((await self.sql("SELECT sub_status FROM credits"))[0][0],"active")

    async def test_late_deleted_event_uses_current_canonical_status(self):
        await self.event("customer.subscription.deleted",{"id":"sub_current","customer":"cus_test","status":"canceled"})
        self.assertEqual((await self.sql("SELECT sub_status FROM credits"))[0][0],"active")

    async def test_real_cancellation_syncs_status_without_removing_credits(self):
        await self.sql("UPDATE credits SET balance=42")
        canceled={**self.sub,"status":"canceled"}
        await self.event("customer.subscription.deleted",{"id":"sub_current","customer":"cus_test"},canceled)
        self.assertEqual((await self.sql("SELECT sub_status FROM credits"))[0][0],"canceled")
        self.assertEqual(await self.balance(),42)

    async def test_previous_period_invoice_does_not_reset_current_allowance(self):
        await self.sql("UPDATE credits SET balance=83")
        sub={**self.sub,"current_period_start":1000,"current_period_end":2000}
        invoice=self.invoice();invoice['lines']={'data':[{'period':{'start':100,'end':1000},'proration':False}]}
        result=await self.event("invoice.paid",invoice,sub)
        self.assertEqual(result['action'],'stale_invoice_ignored')
        self.assertEqual(await self.balance(),83)

    async def test_unscoped_paid_invoice_does_not_change_current_balance(self):
        invoice=self.invoice();invoice.pop("subscription")
        await self.sql("UPDATE credits SET balance=77")
        result=await self.event("invoice.paid",invoice)
        self.assertEqual(result["action"],"unscoped_invoice_ignored")
        self.assertEqual(await self.balance(),77)

    async def test_canceled_subscription_cannot_be_activated_by_late_checkout(self):
        result=await self.event("checkout.session.completed",self.checkout(),{**self.sub,"id":"sub_new","status":"canceled"})
        self.assertEqual(result["action"],"inactive_subscription_ignored")
        self.assertEqual(await self.balance(),0)

    async def test_current_active_plan_cannot_start_a_second_subscription(self):
        with patch.object(billing.stripe.Subscription,"retrieve",return_value=self.sub), patch.object(billing,"_get_or_create_customer",new_callable=AsyncMock) as customer, patch.object(billing,"_create_checkout_session_with_fallback") as checkout:
            with self.assertRaisesRegex(ValueError,"Manage subscription"):
                await billing.create_checkout_session(1,"test@example.invalid","creator")
            customer.assert_not_called();checkout.assert_not_called()

    async def test_past_due_plan_uses_existing_billing_instead_of_duplicate_checkout(self):
        with patch.object(billing.stripe.Subscription,"retrieve",return_value={**self.sub,"status":"past_due"}), patch.object(billing,"_create_checkout_session_with_fallback") as checkout:
            with self.assertRaisesRegex(ValueError,"Manage subscription"):
                await billing.create_checkout_session(1,"test@example.invalid","creator")
            checkout.assert_not_called()

    async def test_previously_canceled_client_can_subscribe_again(self):
        from types import SimpleNamespace
        plans=copy.deepcopy(billing.PLANS);plans["creator"]["prices"]["monthly"]="price_test"
        with patch.object(billing,"PLANS",plans), patch.object(billing.stripe.Subscription,"retrieve",return_value={**self.sub,"status":"canceled"}), patch.object(billing,"_get_or_create_customer",new_callable=AsyncMock,return_value="cus_test"), patch.object(billing,"_create_checkout_session_with_fallback",return_value=SimpleNamespace(url="https://checkout.example.invalid/test")):
            self.assertEqual(await billing.create_checkout_session(1,"test@example.invalid","creator"),"https://checkout.example.invalid/test")

    async def test_unrelated_customer_checkout_cannot_rebind_account(self):
        data=self.checkout("pack_20");data["customer"]="cus_unrelated"
        result=await self.event("checkout.session.completed",data)
        self.assertEqual(result["action"],"customer_mismatch_ignored")
        self.assertEqual(await self.balance(),0)
        self.assertEqual((await self.sql("SELECT stripe_customer_id FROM users"))[0][0],"cus_test")

    async def test_duplicate_affiliate_invoice_cannot_record_a_second_payout(self):
        import time
        await self.sql("INSERT INTO affiliate_commissions(affiliate_email,affiliate_name,affiliate_code,customer_id,started_at) VALUES('affiliate@example.invalid','Test','code','cus_test',?)",(time.time(),))
        await asyncio.gather(*(billing._record_affiliate_payout('cus_test',999,'in_same') for _ in range(4)))
        self.assertEqual((await self.sql('SELECT COUNT(*) FROM affiliate_payouts'))[0][0],1)


if __name__=='__main__':unittest.main()
