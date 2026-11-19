# December 2026 credit gifts

The campaign is installed disabled, with zero enrolled recipients. Deploying code does not approve or grant gifts.

- Redemption window: December 1, 2026 00:00 to January 1, 2027 00:00 exclusive, Europe/Paris.
- Amounts: free 15, Creator 25, Pro 100, Studio 300. Paid amounts are 25% of ONE MONTH, including annual subscriptions and former subscribers.
- No subscription extension, price change, invoice adjustment or extra email.
- Each recipient must be individually approved using a reviewed private manifest containing user ID, email and amount. Never commit that manifest or expose it through an endpoint.
- Before enrollment, take a coherent SQLite backup and check current IDs/emails. Set an auditable approval_reference, insert only the approved recipients, then enable the campaign. Changes to a manifest require fresh owner review.
- An authenticated claim adds credits only when the campaign is enabled, the time window is open, and the current user's ID/email match an approved recipient. The client cannot choose an amount or another account.
- SQLite IMMEDIATE transactions, a unique recipient key and the credit_operations ledger prevent duplicate delivery. Claimed gifts remain in the balance after the redemption window closes.
- Monthly/annual renewal, free daily resets and trial expiry retain unspent gifts. Existing plan access rules are unchanged; free gifts apply to the non-AI tools already included in the free plan.
- GET /api/promotions/christmas-2026 is public without recipient details. Authenticated callers receive only their own eligibility/receipt. Responses are not cached.
- POST /api/promotions/christmas-2026/claim requires authentication. No endpoint enrolls recipients or enables the campaign.

Validation: python -m unittest -q test_annual_credits test_billing_lifecycle test_holiday_credits
