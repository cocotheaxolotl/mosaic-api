"""
Stripe integration — Checkout, Customer Portal, and Webhook handling.
"""

import os
import json

import stripe

from database import get_db
import credits as credits_module

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
SITE_URL = os.environ.get("SITE_URL", "https://cocotheaxolotl.org")

# Plan configuration — Stripe Price IDs set via env vars
PLANS = {
    "creator": {
        "prices": {
            "monthly": os.environ.get("STRIPE_PRICE_CREATOR_MONTHLY", ""),
            "annual": os.environ.get("STRIPE_PRICE_CREATOR_ANNUAL", ""),
            "monthly_eur": os.environ.get("STRIPE_PRICE_CREATOR_MONTHLY_EUR", ""),
            "annual_eur": os.environ.get("STRIPE_PRICE_CREATOR_ANNUAL_EUR", ""),
        },
        "credits": 100,
        "label": "Creator",
        "price_display": "$15",
    },
    "pro": {
        "prices": {
            "monthly": os.environ.get("STRIPE_PRICE_PRO_MONTHLY", ""),
            "annual": os.environ.get("STRIPE_PRICE_PRO_ANNUAL", ""),
            "monthly_eur": os.environ.get("STRIPE_PRICE_PRO_MONTHLY_EUR", ""),
            "annual_eur": os.environ.get("STRIPE_PRICE_PRO_ANNUAL_EUR", ""),
        },
        "credits": 400,
        "label": "Pro",
        "price_display": "$39",
    },
    "studio": {
        "prices": {
            "monthly": os.environ.get("STRIPE_PRICE_STUDIO_MONTHLY", ""),
            "annual": os.environ.get("STRIPE_PRICE_STUDIO_ANNUAL", ""),
            "monthly_eur": os.environ.get("STRIPE_PRICE_STUDIO_MONTHLY_EUR", ""),
            "annual_eur": os.environ.get("STRIPE_PRICE_STUDIO_ANNUAL_EUR", ""),
        },
        "credits": 1500,
        "label": "Studio",
        "price_display": "$79",
    },
}

# Credit packs — one-time purchases
CREDIT_PACKS = {
    "pack_20": {
        "price_id": os.environ.get("STRIPE_PRICE_PACK_20", ""),
        "price_id_eur": os.environ.get("STRIPE_PRICE_PACK_20_EUR", ""),
        "credits": 20,
        "label": "20 Extra Credits",
        "price_display": "$4.99",
    },
    "pack_100": {
        "price_id": os.environ.get("STRIPE_PRICE_PACK_100", ""),
        "price_id_eur": os.environ.get("STRIPE_PRICE_PACK_100_EUR", ""),
        "credits": 100,
        "label": "100 Extra Credits",
        "price_display": "$19.99",
    },
}


def get_plans_public():
    """Return plan info suitable for the pricing page (no secret IDs)."""
    return [
        {
            "key": "free",
            "label": "Free",
            "price_display": "$0",
            "credits": 5,
            "period": "forever",
        },
    ] + [
        {
            "key": key,
            "label": plan["label"],
            "price_display": plan["price_display"],
            "credits": plan["credits"],
            "period": "month",
        }
        for key, plan in PLANS.items()
    ]


async def _get_or_create_customer(user_id: int, email: str) -> str:
    """Get existing Stripe customer ID or create one."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT stripe_customer_id FROM users WHERE id = ?", (user_id,)
        )
        if rows and rows[0][0]:
            return rows[0][0]

        # Create new Stripe customer
        customer = stripe.Customer.create(
            email=email,
            metadata={"coco_user_id": str(user_id)},
        )
        await db.execute(
            "UPDATE users SET stripe_customer_id = ? WHERE id = ?",
            (customer.id, user_id),
        )
        await db.commit()
        return customer.id
    finally:
        await db.close()


async def create_checkout_session(
    user_id: int, email: str, plan_key: str, billing: str = "monthly", ref: str = "", currency: str = "usd"
) -> str:
    """Create a Stripe Checkout session. Returns the checkout URL.
    plan_key: 'creator' | 'pro' | 'studio' | 'pack_20' | 'pack_100'
    billing: 'monthly' | 'annual' (ignored for credit packs)
    currency: 'usd' | 'eur'
    """
    customer_id = await _get_or_create_customer(user_id, email)
    affiliate_ref = ref.strip().lower() if ref else ""

    # Store affiliate_ref on Stripe customer (write-once)
    if affiliate_ref:
        try:
            cust = stripe.Customer.retrieve(customer_id)
            if not (cust.get("metadata") or {}).get("affiliate_ref"):
                stripe.Customer.modify(customer_id, metadata={"affiliate_ref": affiliate_ref})
        except Exception:
            pass

    # Credit pack — one-time payment
    if plan_key in CREDIT_PACKS:
        pack = CREDIT_PACKS[plan_key]
        price_id = pack["price_id_eur"] if currency == "eur" and pack.get("price_id_eur") else pack["price_id"]
        if not price_id:
            raise ValueError(f"Stripe price not configured for pack: {plan_key}")
        session = stripe.checkout.Session.create(
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            mode="payment",
            allow_promotion_codes=True,
            success_url=f"{SITE_URL}/pricing/?success=true&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{SITE_URL}/pricing/?canceled=true",
            metadata={
                "coco_user_id": str(user_id),
                "pack": plan_key,
                "credits": str(pack["credits"]),
                **({"affiliate_ref": affiliate_ref} if affiliate_ref else {}),
            },
        )
        return session.url

    # Subscription plan
    if plan_key not in PLANS:
        raise ValueError(f"Unknown plan: {plan_key}")

    plan = PLANS[plan_key]
    if billing not in ("monthly", "annual"):
        billing = "monthly"
    billing_key = f"{billing}_eur" if currency == "eur" else billing
    price_id = plan["prices"].get(billing_key) or plan["prices"].get(billing, "")
    if not price_id:
        raise ValueError(f"Stripe price not configured for {plan_key}/{billing}")

    session = stripe.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        allow_promotion_codes=True,
        success_url=f"{SITE_URL}/pricing/?success=true&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{SITE_URL}/pricing/?canceled=true",
        metadata={
            "coco_user_id": str(user_id),
            "plan": plan_key,
            **({"affiliate_ref": affiliate_ref} if affiliate_ref else {}),
        },
    )
    return session.url


async def create_portal_session(user_id: int) -> str:
    """Create a Stripe Customer Portal session. Returns the portal URL."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT stripe_customer_id FROM users WHERE id = ?", (user_id,)
        )
        if not rows or not rows[0][0]:
            raise ValueError("No Stripe customer found for this user")

        session = stripe.billing_portal.Session.create(
            customer=rows[0][0],
            return_url=f"{SITE_URL}/pricing/",
        )
        return session.url
    finally:
        await db.close()


async def _find_user_by_customer(customer_id: str) -> int | None:
    """Find user_id by Stripe customer ID."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id FROM users WHERE stripe_customer_id = ?", (customer_id,)
        )
        return rows[0][0] if rows else None
    finally:
        await db.close()


async def _record_affiliate_commission(affiliate_code: str, customer_id: str, subscription_id: str | None, plan: str):
    """Mark affiliate code as used and create commission record (called once per new subscription)."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT id, affiliate_name, affiliate_email, is_used FROM affiliate_codes WHERE code = ? COLLATE NOCASE",
            (affiliate_code,),
        )
        if not rows:
            return  # not a known affiliate code, ignore
        row = rows[0]
        if row[3]:  # is_used
            return  # code already consumed

        # Mark code as used
        import time as _time
        await db.execute(
            "UPDATE affiliate_codes SET is_used = 1, used_at = ? WHERE id = ?",
            (_time.time(), row[0]),
        )
        # Create commission record
        await db.execute(
            """INSERT INTO affiliate_commissions
               (affiliate_email, affiliate_name, affiliate_code, customer_id, subscription_id, plan)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row[2], row[1], affiliate_code, customer_id, subscription_id, plan),
        )
        await db.commit()
    finally:
        await db.close()


async def _record_affiliate_payout(customer_id: str, amount_cents: int, invoice_id: str | None):
    """If an active commission exists for this customer and within 6 months, record a 30% payout."""
    import time as _time
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            """SELECT id, affiliate_email, commission_rate, months_total, months_paid, started_at
               FROM affiliate_commissions
               WHERE customer_id = ? AND status = 'active'""",
            (customer_id,),
        )
        if not rows:
            return
        comm = rows[0]
        comm_id, aff_email, rate, months_total, months_paid, started_at = comm

        # Check we haven't exceeded months_total
        if months_paid >= months_total:
            await db.execute(
                "UPDATE affiliate_commissions SET status = 'completed' WHERE id = ?", (comm_id,)
            )
            await db.commit()
            return

        # Also check 6-month wall-clock guard
        months_elapsed = (_time.time() - started_at) / (30 * 86400)
        if months_elapsed > months_total + 1:
            await db.execute(
                "UPDATE affiliate_commissions SET status = 'completed' WHERE id = ?", (comm_id,)
            )
            await db.commit()
            return

        amount_usd = round(amount_cents / 100 * rate, 2)
        new_months_paid = months_paid + 1

        await db.execute(
            """INSERT INTO affiliate_payouts
               (affiliate_email, commission_id, month_number, amount_usd, invoice_id)
               VALUES (?, ?, ?, ?, ?)""",
            (aff_email, comm_id, new_months_paid, amount_usd, invoice_id),
        )
        await db.execute(
            "UPDATE affiliate_commissions SET months_paid = ?, status = ? WHERE id = ?",
            (new_months_paid, 'completed' if new_months_paid >= months_total else 'active', comm_id),
        )
        await db.commit()
    finally:
        await db.close()


async def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """Process a Stripe webhook event. Returns a status dict."""
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise ValueError("Invalid webhook signature")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id_str = data.get("metadata", {}).get("coco_user_id")
        customer_id = data.get("customer")

        # Credit pack purchase (one-time payment)
        pack_key = data.get("metadata", {}).get("pack")
        if user_id_str and pack_key and pack_key in CREDIT_PACKS:
            user_id = int(user_id_str)
            pack = CREDIT_PACKS[pack_key]
            await credits_module.add_credits(
                user_id,
                amount=pack["credits"],
                reason="credit_pack_purchase",
                metadata=json.dumps({"pack": pack_key}),
            )
            return {"handled": True, "action": "credit_pack_purchased", "pack": pack_key, "credits": pack["credits"]}

        # New subscription activated
        plan_key = data.get("metadata", {}).get("plan")
        if user_id_str and plan_key and plan_key in PLANS:
            user_id = int(user_id_str)
            plan = PLANS[plan_key]
            sub_id = data.get("subscription")
            affiliate_ref = (data.get("metadata", {}).get("affiliate_ref") or "").strip().lower()

            # Update user's Stripe customer ID if needed
            db = await get_db()
            try:
                await db.execute(
                    "UPDATE users SET stripe_customer_id = ? WHERE id = ?",
                    (customer_id, user_id),
                )
                await db.commit()
            finally:
                await db.close()

            # Set plan and credit the account
            await credits_module.set_plan(
                user_id,
                plan_name=plan_key,
                monthly_quota=plan["credits"],
                stripe_sub_id=sub_id,
                sub_status="active",
            )

            # Record affiliate commission if a valid code was used
            if affiliate_ref:
                await _record_affiliate_commission(
                    affiliate_code=affiliate_ref,
                    customer_id=customer_id,
                    subscription_id=sub_id,
                    plan=plan_key,
                )

            return {"handled": True, "action": "subscription_started", "plan": plan_key}

    elif event_type == "invoice.paid":
        # Monthly renewal — reset credits
        customer_id = data.get("customer")
        if customer_id:
            user_id = await _find_user_by_customer(customer_id)
            if user_id:
                # Only renew if this is not the first invoice (checkout.session.completed handles that)
                billing_reason = data.get("billing_reason")
                if billing_reason == "subscription_cycle":
                    await credits_module.renew_credits(user_id)
                    # Record affiliate payout if there is an active commission
                    amount_paid = data.get("amount_paid", 0)  # in cents
                    invoice_id = data.get("id")
                    await _record_affiliate_payout(customer_id, amount_paid, invoice_id)
                    return {"handled": True, "action": "credits_renewed"}

    elif event_type == "customer.subscription.updated":
        customer_id = data.get("customer")
        sub_status = data.get("status")
        if customer_id:
            user_id = await _find_user_by_customer(customer_id)
            if user_id:
                db = await get_db()
                try:
                    await db.execute(
                        "UPDATE credits SET sub_status = ? WHERE user_id = ?",
                        (sub_status, user_id),
                    )
                    await db.commit()
                finally:
                    await db.close()
                return {"handled": True, "action": "subscription_updated", "status": sub_status}

    elif event_type == "customer.subscription.deleted":
        customer_id = data.get("customer")
        if customer_id:
            user_id = await _find_user_by_customer(customer_id)
            if user_id:
                await credits_module.cancel_plan(user_id)
                return {"handled": True, "action": "subscription_canceled"}

    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer")
        if customer_id:
            user_id = await _find_user_by_customer(customer_id)
            if user_id:
                db = await get_db()
                try:
                    await db.execute(
                        "UPDATE credits SET sub_status = 'past_due' WHERE user_id = ?",
                        (user_id,),
                    )
                    await db.commit()
                finally:
                    await db.close()
                return {"handled": True, "action": "payment_failed"}

    return {"handled": False, "event_type": event_type}
