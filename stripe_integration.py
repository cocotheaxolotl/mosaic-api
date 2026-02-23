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
        },
        "credits": 100,
        "label": "Creator",
        "price_display": "$9.99",
    },
    "pro": {
        "prices": {
            "monthly": os.environ.get("STRIPE_PRICE_PRO_MONTHLY", ""),
            "annual": os.environ.get("STRIPE_PRICE_PRO_ANNUAL", ""),
        },
        "credits": 400,
        "label": "Pro",
        "price_display": "$24.99",
    },
    "studio": {
        "prices": {
            "monthly": os.environ.get("STRIPE_PRICE_STUDIO_MONTHLY", ""),
            "annual": os.environ.get("STRIPE_PRICE_STUDIO_ANNUAL", ""),
        },
        "credits": 1500,
        "label": "Studio",
        "price_display": "$49.99",
    },
}

# Credit packs — one-time purchases
CREDIT_PACKS = {
    "pack_20": {
        "price_id": os.environ.get("STRIPE_PRICE_PACK_20", ""),
        "credits": 20,
        "label": "20 Extra Credits",
        "price_display": "$4.99",
    },
    "pack_100": {
        "price_id": os.environ.get("STRIPE_PRICE_PACK_100", ""),
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
    user_id: int, email: str, plan_key: str, billing: str = "monthly", ref: str = ""
) -> str:
    """Create a Stripe Checkout session. Returns the checkout URL.
    plan_key: 'creator' | 'pro' | 'studio' | 'pack_20' | 'pack_100'
    billing: 'monthly' | 'annual' (ignored for credit packs)
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
        if not pack["price_id"]:
            raise ValueError(f"Stripe price not configured for pack: {plan_key}")
        session = stripe.checkout.Session.create(
            customer=customer_id,
            line_items=[{"price": pack["price_id"], "quantity": 1}],
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
    price_id = plan["prices"].get(billing, "")
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
