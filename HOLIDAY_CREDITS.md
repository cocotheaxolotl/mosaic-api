# December 2026 tool-credit gifts (no paid AI)

The campaign is disabled by default with zero enrolled recipients. Deployment does not authorize or grant gifts.

- Redemption: December 1, 2026 00:00 to January 1, 2027 00:00 exclusive, Europe/Paris.
- Amounts: free 15, Creator 25, Pro 100, Studio 300. Paid quantities are 25% of ONE MONTH even for annual and former subscriptions.
- Gifts go only to tool_credit_balances, with tool_credit_transactions and a unique credit_operations receipt. They NEVER increase credits.balance, monthly_quota, subscription dates or paid AI allowance.
- No automatic enrollment, extension, invoice, charge, price change or extra email.
- Approved browser exports: maze, sudoku, wordsearch, crossword, bingo, cryptogram, handwriting, mathdrills, back-matter, book-template, qr-code. One credit per export. Unknown or AI routes cannot spend this wallet.
- POST /api/credits/tools/consume uses daily free allowance first for free users, then tool gifts. Paid users spend tool gifts before their ordinary balance. Without gifts/for other tools, the normal one-credit export behavior remains.
- Paid APIs continue to consume only credits.balance. GET /api/credits/balance and /api/auth/me return the separate tool_credits field without mixing balances. The image AI endpoint rejects free accounts before a provider call. The ordinary consume endpoint enforces AI image quality costs 1/3/10.
- Tool gifts survive renewal, upgrade, trial expiry and the end of the redemption window. Standard existing packs and customer-service gifts keep their original protections.
- GET /api/promotions/christmas-2026 exposes only public dates/type, plus a logged-in user's own approved eligibility. Cache-Control: no-store. POST .../claim checks dates, campaign approval, exact user/email and one-time delivery under SQLite IMMEDIATE transaction.
- An individually reviewed private manifest must state credit_type=non_api and list each ID/email/amount. Before activation: verify the named list, take a coherent SQLite backup, enroll only approved IDs, set an auditable approval reference. Never commit the manifest or add a public enrollment endpoint.
- Newly registered accounts are not implicitly added to an older approved list; additions require review.

Tool gift usage makes no paid provider call. This does not guarantee an unchanged total business bill: hosting and ordinary AI usage remain separate.

Validation: python -m unittest -q test_annual_credits test_billing_lifecycle test_holiday_credits
