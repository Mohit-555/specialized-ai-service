# Annotation Policy — AI Classification Service V1

## Intent Definitions

### general_question
A broad question about the company, product, service, or process that does not fit into a more specific category.
- "What are your business hours?"
- "Do you have a mobile app?"
- "How do I sign up?"

### product_question
A specific question about features, functionality, compatibility, or specifications of a product/service.
- "Does your premium plan include API access?"
- "Can I integrate with Salesforce?"

### pricing
Any question or statement about cost, fees, plans, billing amounts, or price comparison.
- "How much does the pro plan cost?"
- "Is there a discount for annual billing?"

### sales
Expressing intent to purchase, upgrade, or showing buying signals. Includes feature requests framed as purchase intent.
- "I'd like to upgrade to the enterprise plan."
- "Can I get a demo before purchasing?"

### technical_support
Reporting a technical problem, bug, error, or malfunction that prevents normal usage.
- "The dashboard keeps showing a 500 error."
- "My reports won't export to PDF."

### complaint
Expressing dissatisfaction, frustration, or negative experience without explicitly demanding a refund. General venting/negative feedback.
- "Your service is terrible. I'm very disappointed."
- "This is the worst customer experience I've had."

### refund
Explicit demand or request for a refund, money back, or cancellation tied to reimbursement.
- "I want my money back."
- "Refund my payment immediately."

### account_issue
Problems related to account access, login, profile, verification, or account-specific settings.
- "I can't log into my account."
- "My email was changed without my permission."

### human_request
Explicit request to speak with a human agent, representative, or real person. Note: frustration alone does not qualify.
- "Can I speak to a real person?"
- "Human se baat karni hai."

### other
Any message that does not fit any of the above intents. Greetings, chit-chat, gibberish, spam, out-of-scope questions.
- "Hello"
- "What's the weather today?"

---

## Intent Boundary Rules

### human_request vs sales
→ If someone asks to speak to sales/sales team → `sales`, not `human_request`.
→ `human_request` is reserved for requests to speak to **any** human, typically when frustrated with automation.

### human_request vs complaint
→ A complaint without an explicit "talk to a person" request → `complaint`.
→ Only if they explicitly ask to speak to a human → `human_request`.
→ "I am very angry. Talk to someone." → `human_request` only if the request is clear.

### complaint vs refund
→ If the user demands money back → `refund`.
→ If they express dissatisfaction without demanding money → `complaint`.
→ "Your service is useless" → `complaint`.
→ "I want a full refund immediately" → `refund`.
→ "I'm unhappy with the service and I'd like a refund please" → `refund` (refund intent dominates).

### account_issue vs complaint
→ If the primary issue is account-specific (login, access, profile) → `account_issue`.
→ If they mainly express frustration but mention account words incidentally → `complaint`.
→ "I can't access my account and I'm furious" → `account_issue` (primary issue is account access).

### account_issue vs pricing
→ If they can't access billing/pricing page → `account_issue`.
→ If they are asking about cost → `pricing`.
→ "I can't see my plan pricing page" → `account_issue`.
→ "Why is my plan showing a different price?" → `pricing`.

### product_question vs general_question
→ If the question is about a specific product feature/function → `product_question`.
→ If it's about general company info → `general_question`.
→ "Does your tool support Python?" → `product_question`.
→ "What does your company do?" → `general_question`.

### sales vs pricing
→ If they are asking "how much" → `pricing`.
→ If they express desire to buy/upgrade → `sales`.
→ "I want to buy the premium plan" → `sales`.
→ "What's the price of premium?" → `pricing`.

### other vs general_question
→ If the question is on-topic but broad → `general_question`.
→ If completely off-topic or gibberish → `other`.

---

## Multi-Intent Messages

Each message gets exactly ONE primary intent.

Selection rules:
1. If a refund is demanded → `refund` (highest priority).
2. If human contact is explicitly requested → `human_request`.
3. If a technical problem is described → `technical_support`.
4. If account access is the core issue → `account_issue`.
5. If complaint language dominates → `complaint`.
6. Purchase intent → `sales`.
7. Pricing question → `pricing`.
8. Product feature question → `product_question`.
9. Broad on-topic question → `general_question`.
10. Everything else → `other`.

---

## Escalation Labeling Rules

### escalation_required = true
- Customer explicitly asks to speak to a human.
- Customer demands immediate action (refund/account fix) with urgency/anger.
- Threat of legal action, chargeback, or regulatory complaint.
- Repeated frustration indicating self-service has failed.
- Account security concerns (unauthorized access, hacked).

### escalation_required = false
- Simple questions or information requests.
- Routine feature inquiries.
- Users indicating they do NOT need human help (even if angry).
- Messages containing escalation keywords in a non-escalation context.
  - "I don't want a refund, I just want to know your policy."
  - "Nobody needs to contact me, I just have a question."
- Resolved issues: "I was angry earlier but it's fine now."
- Pure feedback without action requested.

### Hard Negative Examples (keyword ≠ escalation)
- "I don't want a refund." → escalation_required = false
- "I was angry but it's resolved." → escalation_required = false
- "Just explain your refund policy." → escalation_required = false
- "Your service is bad but I don't need anyone to call." → escalation_required = false

### Hard Positive Examples (no obvious keyword but escalation)
- "I've been waiting three weeks for a resolution." → escalation_required = true
- "This is the third time I'm explaining the same issue." → escalation_required = true
- "I'm going to dispute this with my bank." → escalation_required = true
