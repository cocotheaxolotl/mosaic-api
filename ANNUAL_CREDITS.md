# Monthly credits for annual subscriptions

Annual subscriptions are invoiced once a year. The background worker in
`annual_credits.py` grants their monthly entitlement independently of invoices,
within five minutes of the monthly UTC anniversary. It uses the current paid
annual term from Stripe (legacy subscription fields or current item fields).
Canceled, unpaid, expired, unlinked and unsupported subscriptions are excluded.
The existing monthly invoice path is retained, with a guard against an older
subscription changing the balance of the customer's current subscription.

`annual_credit_periods` has a database primary key on subscription and period.
Balance changes and the period record commit together under `BEGIN IMMEDIATE`.
Consumption and purchases take the same write lock. This protects retries,
parallel workers and concurrent user operations. Creator/Pro refresh their
monthly quota and keep unspent packs. Studio retains its balance and adds the
monthly quota, including due periods missed during an outage. There is no
external scheduler to configure; FastAPI starts and stops one worker.

## Existing missed periods require individual approval

The first database initialization of this release persists a single
`annual_credit_policy.live_from` timestamp. It is never advanced on restart.
An unallocated period starting before that cutoff is logged `pending_review`:
no balance change is made. Later periods follow the normal automatic rules.
Existing initial grants and yearly invoice grants are recognized so they are
not granted a second time. Old non-rollover periods after the cutoff that have
already expired are recorded without being accumulated as compensation.

Before any approved manual correction, re-read the current balance, matching
subscription and period, and apply the balance change, audit transaction and
period status in one transaction. Mark the same ledger period `granted` so a
worker retry cannot undo or duplicate the correction. Record the quota and
period in transaction metadata; the transaction delta is the actual change.
Do not delete pending entries or change the policy cutoff to force a backfill.
Do not run historical backfills from startup or an unauthenticated endpoint.

## Validation and deployment

Run `python -m unittest -v test_annual_credits` and compile the changed modules.
The tests use temporary SQLite databases and fake Stripe responses. They never
send mail, charge customers or mutate production data.

On deployment, retain the existing Fly volume and machine configuration.
The migration only adds its two ledger/policy tables. Check `/api/health`,
the live source hashes and the ledger status counts. A pending historical
period is not a repaired customer balance: verify corrections separately after
explicit customer-by-customer approval. Avoid logging tokens or publishing
client-level audit exports in this repository.

Builds use `constraints-production.txt`, captured from the working production
image, so a credit fix does not silently upgrade unrelated runtime packages.
`httpx` is a direct application dependency. Docker imports the app and runs the
offline credit regression suite before an image can be rolled out.

Approved service gifts use the `customer_service_gift` transaction reason.
They remain separate from the monthly entitlement and are protected like packs
at annual-plan monthly renewals. Subscription credits are consumed first.

## Payment lifecycle protections

`credit_operations` records paid Checkout sessions and invoice renewals in the
same transaction as their credit changes. A duplicated completed/async event
cannot fulfill the same session twice. Legacy subscription starts and current
monthly grants are recognized when bootstrapping this ledger.

Monthly invoice renewals retain packs and service gifts; Studio retains its
unused subscription allowance as well. Old invoices cannot renew a later
period, and invoice/state events for another subscription of the same customer
cannot modify the current contract. Status events fetch Stripe's canonical
state so delayed failures or cancellations cannot overwrite a newer state.
These guards never cancel, refund or change a Stripe subscription.

The production webhook must enable `checkout.session.async_payment_succeeded`
and `checkout.session.async_payment_failed` in addition to the existing events.
A completed but unpaid Checkout session waits for payment success. After a
successful late payment, the session is fulfilled once. Do not replay old paid
events as an audit: inspect invoices and the ledger in read-only mode instead.
