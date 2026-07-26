import csv
import random
import json
import os

random.seed(42)

V1_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v1")
V3_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v3")
os.makedirs(V3_DIR, exist_ok=True)

INTENTS = [
    "general_question", "product_question", "pricing", "sales",
    "technical_support", "complaint", "refund", "account_issue",
    "human_request", "other"
]

examples = []

def add(text, intent, escalation, tags=None, note=""):
    examples.append({
        "text": text,
        "intent": intent,
        "escalation_required": "true" if escalation else "false",
        "language_style": "",
        "difficulty": "",
        "scenario_type": "",
        "notes": note,
        "tags": json.dumps(tags or [])
    })

def batch_add(items):
    for item in items:
        if len(item) == 4:
            text, intent, escalation, tags = item
            add(text, intent, escalation, tags, "")
        elif len(item) == 5:
            text, intent, escalation, tags, note = item
            add(text, intent, escalation, tags, note)

# ============================================================
# LOAD ALL EXISTING V1 EXAMPLES
# ============================================================
v1_csv = os.path.join(V1_DIR, "dataset_v1_raw.csv")
v1_count = 0
if os.path.exists(v1_csv):
    with open(v1_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(dict(row))
            v1_count += 1

print(f"Loaded {v1_count} examples from dataset-v1")

# ============================================================
# 1. CONFUSION PAIR CURRICULUM (~400 new)
# ============================================================

# product_question vs general_question
batch_add([
    ("Does the premium plan support unlimited users?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("What kind of products do you offer?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Can I integrate your tool with my existing CRM?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Do you have a CRM tool?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Is end-to-end encryption available in the basic plan?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("What security measures do you have?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Does your pro plan support webhook retries?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("How does your service handle security?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Can I set up custom automations with the API?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Do you have automation features in your product?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
    ("What is the maximum file size for attachments in the premium tier?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("What are your file upload limits?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Does your platform support custom user roles and permissions?", "product_question", False, ["standard"], "v3 confusion: product vs general"),
    ("Can I control who sees what in my account?", "general_question", False, ["standard"], "v3 confusion: product vs general"),
])

# technical_support vs complaint (14 total errors in analysis)
batch_add([
    ("My reports are generating errors every time I export", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Your reports feature is useless and broken", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("I keep getting kicked out of the dashboard mid-session", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Your platform keeps logging me out and it's infuriating", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("The export function throws a 500 error every single time", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Your export function is a complete disaster", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Every time I try to save my work the page crashes", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("I cannot believe how unstable this software is", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("The integration keeps dropping connection every 10 minutes", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Your integrations are unreliable and frustrating", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("My data sync has been stuck at 90% for two hours", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Nothing ever syncs properly on this platform", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("I get a blank white screen when I open the analytics tab", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Your analytics page is completely broken just like everything else", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Can you fix the mobile app? It closes immediately on launch", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("Your mobile app is garbage. It never works.", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    # Mixed where both tech and complaint are present:
    ("The dashboard is glitching and I'm really annoyed by it", "technical_support", False, ["standard", "multi_intent"], "v3 confusion: tech vs complaint mixed"),
    ("I'm annoyed that the dashboard keeps glitching, please fix it", "technical_support", False, ["standard"], "v3 confusion: tech vs complaint mixed"),
    ("Your software has so many bugs I want to switch providers", "complaint", False, ["standard"], "v3 confusion: tech vs complaint"),
    ("There is a critical bug in the payment module that needs fixing", "technical_support", True, ["standard"], "v3 confusion: tech vs complaint"),
])

# pricing vs sales
batch_add([
    ("How much does the premium plan cost?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("I want to buy the premium plan", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("Can you tell me the price of enterprise?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("I'd like to get started with the enterprise plan", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("What is the monthly subscription fee?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("Please start my monthly subscription now", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("How much per user annually?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("We want to sign up 50 users on the annual plan", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("Is there a setup fee or is it included?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("We are ready to purchase. Send me the contract.", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("What discounts are available for annual commitments?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("I want to commit to an annual plan. Process my payment.", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("Can I get a breakdown of all the costs involved?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("Please invoice me for the enterprise tier upgrade", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("How does your pricing compare to competitors?", "pricing", False, ["standard"], "v3 confusion: pricing vs sales"),
    ("I'm switching from a competitor. Help me set up my account.", "sales", False, ["standard"], "v3 confusion: pricing vs sales"),
])

# human_request vs sales
batch_add([
    ("I need to speak to a live person now", "human_request", True, ["standard"], "v3 confusion: human vs sales"),
    ("I want to speak to the sales team about purchasing", "sales", False, ["standard"], "v3 confusion: human vs sales"),
    ("Get me a human on the line immediately", "human_request", True, ["standard"], "v3 confusion: human vs sales"),
    ("Please connect me to a sales representative", "sales", False, ["standard"], "v3 confusion: human vs sales"),
    ("I need to talk to a real person about my problem", "human_request", True, ["standard"], "v3 confusion: human vs sales"),
    ("I want to discuss purchasing options with a real person", "sales", False, ["standard"], "v3 confusion: human vs sales"),
    ("Stop the chatbot. Give me a human agent.", "human_request", True, ["standard"], "v3 confusion: human vs sales"),
    ("Connect me to your sales department for a demo", "sales", False, ["standard"], "v3 confusion: human vs sales"),
    ("Your bot is useless. I want to speak to someone real.", "human_request", True, ["standard"], "v3 confusion: human vs sales"),
    ("I need a walkthrough from the sales engineering team", "sales", False, ["standard"], "v3 confusion: human vs sales"),
])

# general_question vs technical_support
batch_add([
    ("Can I access my account from multiple devices?", "general_question", False, ["standard"], "v3 confusion: general vs tech"),
    ("I cannot access my account from any device. It says forbidden.", "technical_support", False, ["standard"], "v3 confusion: general vs tech"),
    ("How do I reset my password?", "general_question", False, ["standard"], "v3 confusion: general vs tech"),
    ("The password reset link is broken. Help.", "technical_support", False, ["standard"], "v3 confusion: general vs tech"),
    ("Can I share files with external users?", "general_question", False, ["standard"], "v3 confusion: general vs tech"),
    ("File sharing with external users is throwing an error.", "technical_support", False, ["standard"], "v3 confusion: general vs tech"),
    ("How do I enable two-factor authentication?", "general_question", False, ["standard"], "v3 confusion: general vs tech"),
    ("Two-factor authentication is not sending me the code.", "technical_support", False, ["standard"], "v3 confusion: general vs tech"),
])

# account_issue vs technical_support
batch_add([
    ("I can't log into my account at all", "account_issue", False, ["standard"], "v3 confusion: account vs tech"),
    ("The login page keeps refreshing and never loads", "technical_support", False, ["standard"], "v3 confusion: account vs tech"),
    ("My password was changed without my permission", "account_issue", True, ["standard"], "v3 confusion: account vs tech"),
    ("The forgot password flow fails at the last step", "technical_support", False, ["standard"], "v3 confusion: account vs tech"),
    ("Someone else's data is appearing in my dashboard", "account_issue", True, ["standard"], "v3 confusion: account vs tech"),
    ("The dashboard is showing data from another user's account", "technical_support", True, ["standard"], "v3 confusion: account vs tech"),
    ("My account shows as inactive even though I paid", "account_issue", True, ["standard"], "v3 confusion: account vs tech"),
    ("The payment confirmation page shows an error after success", "technical_support", True, ["standard"], "v3 confusion: account vs tech"),
])

# ============================================================
# 2. ESCALATION HARD NEGATIVES (~300 new)
# ============================================================
batch_add([
    ("I said I wanted a refund earlier but I've changed my mind. Don't process it.", "refund", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I was angry about the billing issue but it's sorted now. No need to escalate.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Please do NOT transfer me to anyone. I just want an answer.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Don't call me about this. Email response is fine.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I don't need human intervention. Your automated response answered my question.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I explicitly do not want a refund. I just want to know the policy.", "general_question", False, ["hard_negative_escalation", "negation"], "v3 esc hard neg"),
    ("Please do not escalate this to any team. I solved it myself.", "technical_support", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Don't connect me to support. I figured out the issue.", "technical_support", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Nobody from your team needs to follow up. I'm all set.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("This is NOT a support request. It's just feedback.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I don't need anyone to reach out. Just acknowledge my message.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("My problem was fixed. Close this request.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("I had an issue but it's resolved now. Thanks.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("Earlier I was frustrated but everything works now. No action needed.", "complaint", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("I don't want compensation. I just wanted to let you know.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Please don't escalate. I just wanted to share my experience.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("No need to call. Just reply here when you can.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I explicitly asked you not to contact me. Just process the request.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Do not escalate. I am happy with the solution provided.", "general_question", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("Forget about it. I handled it differently.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("No action required. Just giving a heads up.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Calm down, it's not that serious. Just tell me the answer.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Please don't send anyone to call me. I work night shifts.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("The issue is already fixed on my end. Thanks.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("I resolved it using the help article. No human needed.", "technical_support", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I know I sounded angry but I don't need any follow-up.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I was about to request a refund but your team fixed it. Thanks!", "refund", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("Don't process anything. I was just asking hypothetically.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Please no calls. SMS or email is better for me.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("This is just a complaint. I don't want anyone to contact me.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I know I mentioned refund but I don't want one. Just explaining my situation.", "complaint", False, ["hard_negative_escalation", "negation"], "v3 esc hard neg"),
    ("I don't want to speak to a manager. You answered my question.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Don't bother anyone about this. I was just venting.", "complaint", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("No escalation warranted. Simple inquiry.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("I do not consent to being contacted. Just reply in this thread.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Not urgent. Just let me know when you have an answer.", "general_question", False, ["hard_negative_escalation"], "v3 esc hard neg"),
    ("Please do not process any refund. I was mistaken.", "refund", False, ["hard_negative_escalation", "negation"], "v3 esc hard neg"),
    ("I already spoke to someone. All good. Close the ticket.", "general_question", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("Cancel my escalation request. Everything is fine now.", "complaint", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("I've sorted it. No need to escalate or call.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 esc hard neg"),
    ("Maine khud fix kar liya. Koi call mat karo.", "technical_support", False, ["hard_negative_escalation", "hinglish"], "v3 esc hard neg hinglish"),
    ("Bas price chahiye tha. Koi escalate mat karo.", "pricing", False, ["hard_negative_escalation", "hinglish"], "v3 esc hard neg hinglish"),
    ("Mera issue resolve ho gaya. Ticket close karo.", "technical_support", False, ["hard_negative_escalation", "resolution_state", "hinglish"], "v3 esc hard neg hinglish"),
    ("Refund nahi chahiye. Bas policy samajhni thi.", "refund", False, ["hard_negative_escalation", "negation", "hinglish"], "v3 esc hard neg hinglish"),
    ("Koi mujhe call karke follow up mat karo. Sab theek hai.", "general_question", False, ["hard_negative_escalation", "resolution_state", "hinglish"], "v3 esc hard neg hinglish"),
])

# ============================================================
# 3. ESCALATION HARD POSITIVES (~300 new)
# ============================================================
batch_add([
    ("I have a deadline in 2 hours and I can't access my files.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("This is the fourth time I'm reporting the same bug.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("My entire team is blocked because of this issue.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Your automated responses aren't helping with this specific situation.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I've tried everything in your knowledge base. Nothing works.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Mera kaam ruka hua hai. Aap log seriously nahi le rahe.", "technical_support", True, ["escalation_positive", "hinglish"], "v3 esc hard pos hinglish"),
    ("Someone logged into my account from a different country.", "account_issue", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I will dispute this with my credit card company.", "refund", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I've lost confidence in your platform after this incident.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
    ("We have a board meeting tomorrow and your system is down.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I need this resolved before the end of the business day.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("My client is waiting for me to deliver a report using your tool.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I've been transferred between three agents with no resolution.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
    ("This is a security vulnerability that needs immediate patching.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I cannot stress enough how urgent this is.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Meri deadline aaj sham tak hai. Please koi senior ko batao.", "technical_support", True, ["escalation_positive", "hinglish"], "v3 esc hard pos hinglish"),
    ("I found a security flaw in your authentication system.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("My entire revenue depends on this tool working properly.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
    ("This is the fifth time I'm raising the same issue. I need a manager.", "human_request", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I've been on hold for 45 minutes. This is unacceptable.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Our quarterly report is due tomorrow and analytics is broken.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I'm going to miss my project deadline because of your downtime.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("There is a data discrepancy that could affect our audit.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Please escalate to your highest level of support. This is critical.", "human_request", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I have evidence of a data breach on your platform.", "account_issue", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Your service failure cost me a client. I need compensation.", "refund", True, ["escalation_positive"], "v3 esc hard pos"),
    ("My account has been compromised and transactions were made.", "account_issue", True, ["escalation_positive"], "v3 esc hard pos"),
    ("This is affecting 200+ users in my organization.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I need an emergency patch deployed to production.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Aapki vajah se mera client loss ho raha hai. Immediate action lo.", "complaint", True, ["escalation_positive", "hinglish"], "v3 esc hard pos hinglish"),
    ("Your system deleted my data without any warning.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("This has been unresolved for 6 days and I'm losing patience.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I'm about to miss a regulatory filing because of your bug.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Your platform went down during my live presentation.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I need someone with authority to override this restriction.", "human_request", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Meri poori team block hai. Koi senior ko escalate karo.", "technical_support", True, ["escalation_positive", "hinglish"], "v3 esc hard pos hinglish"),
    ("This is a production outage. We are losing money every minute.", "technical_support", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I require a written explanation for this billing error.", "pricing", True, ["escalation_positive"], "v3 esc hard pos"),
    ("I will report this to the data protection authority.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
    ("Your negligence caused a leak of my confidential information.", "complaint", True, ["escalation_positive"], "v3 esc hard pos"),
])

# ============================================================
# 4. HINGLISH (+800 new)
# ============================================================
hinglish_intents = [
    # general_question
    ("mera payment ho gaya but account activate nahi hua", "technical_support", True, ["hinglish"]),
    ("refund nahi chahiye bas policy batao", "refund", False, ["hinglish", "negation"]),
    ("koi support person se baat karwa do", "human_request", True, ["hinglish"]),
    ("premium plan ka price kya hai", "pricing", False, ["hinglish"]),
    ("payment deduct ho gaya but service start nahi hui", "technical_support", True, ["hinglish"]),
    ("mujhe apna account delete karna hai", "general_question", False, ["hinglish"]),
    ("Mera issue resolved nahi ho raha hai since 3 days", "technical_support", False, ["hinglish"]),
    ("price kya hai", "pricing", False, ["hinglish"]),
    ("refund chahiye", "refund", True, ["hinglish"]),
    ("human do", "human_request", True, ["hinglish"]),
    ("mujhe premium plan lena hai 50 logon ke liye", "sales", False, ["hinglish"]),
    ("kya isme bulk SMS hai", "product_question", False, ["hinglish"]),
    ("mera email change nahi ho raha account mein", "account_issue", False, ["hinglish"]),
    ("Maine 5 baar complaint kiya hai but koi sun nahi raha", "complaint", True, ["hinglish"]),
    ("Bas price batao, koi call mat karo", "pricing", False, ["hinglish", "hard_negative_escalation"]),
    ("mujhe apna password reset karna hai", "account_issue", False, ["hinglish"]),
    ("aapki app kyu crash ho rahi hai har baar", "technical_support", False, ["hinglish"]),
    ("kya mein apna plan change kar sakta hu", "pricing", False, ["hinglish"]),
    ("mera data export karo", "general_question", False, ["hinglish"]),
    ("aapke paas koi discount code hai kya", "pricing", False, ["hinglish"]),
    ("kya aapke yaha student discount milta hai", "pricing", False, ["hinglish"]),
    ("mujhe apni billing history dekhni hai", "pricing", False, ["hinglish"]),
    ("aapki service band kaise kare", "general_question", False, ["hinglish"]),
    ("mera trial period kab khatam ho raha hai", "general_question", False, ["hinglish"]),
    ("kya aap Hindi mein bhi support dete ho", "general_question", False, ["hinglish"]),
    ("mujhe team plan chahiye 20 logon ke liye", "sales", False, ["hinglish"]),
    ("mera invoice galat hai dobara bhejo", "pricing", False, ["hinglish"]),
    ("aapne mujhe double charge kar diya refund karo", "refund", True, ["hinglish"]),
    ("kya isme auto backup feature hai", "product_question", False, ["hinglish"]),
    ("mujhe koi expert se baat karni hai", "human_request", True, ["hinglish"]),
    ("mera account hacked ho gaya hai help", "account_issue", True, ["hinglish"]),
    ("aapki website down hai kya", "technical_support", True, ["hinglish"]),
    ("mujhe product demo chahiye", "sales", False, ["hinglish"]),
    ("kaise pata kare ki konsa plan sahi hai mere liye", "general_question", False, ["hinglish"]),
    ("aapke software mein kya kya features hain", "product_question", False, ["hinglish"]),
    ("meri report generate nahi ho rahi error aa raha hai", "technical_support", False, ["hinglish"]),
    ("mujhe annual plan lena hai koi discount do", "sales", False, ["hinglish"]),
    ("maine payment kar diya lekin account activate nahi hua", "technical_support", True, ["hinglish"]),
    ("aap mujhe call karo main samjhaunga", "human_request", True, ["hinglish"]),
    ("kya aapke yaha cashback milta hai", "pricing", False, ["hinglish"]),
    ("mera account suspend kyu hua", "account_issue", True, ["hinglish"]),
    ("aapki app bahut slow hai improve karo", "technical_support", False, ["hinglish"]),
    ("mujhe aapka product pasand hai but price zyada hai", "pricing", False, ["hinglish", "multi_intent"]),
    ("kya main apna username change kar sakta hu", "general_question", False, ["hinglish"]),
    ("mera payment method add nahi ho raha", "account_issue", False, ["hinglish"]),
    ("aapki team ka response time bahut slow hai", "complaint", False, ["hinglish"]),
    ("mujhe bulk discount chahiye 100 users ke liye", "sales", False, ["hinglish"]),
    ("kya aap custom integration support karte ho", "product_question", False, ["hinglish"]),
    ("maine apna plan upgrade kiya lekin features unlock nahi hue", "technical_support", False, ["hinglish"]),
    ("mujhe do mahine ka refund chahiye", "refund", True, ["hinglish"]),
    ("aapne mujhe galat plan charge kiya hai", "pricing", True, ["hinglish"]),
    ("kya aapke paas koi referral program hai", "general_question", False, ["hinglish"]),
    ("meri auto-renewal band karo", "sales", False, ["hinglish"]),
    ("mujhe koi SMS notification nahi aa raha", "technical_support", False, ["hinglish"]),
    ("aapka dashboard error de raha hai fix karo", "technical_support", False, ["hinglish"]),
    ("kya main phone se bhi support le sakta hu", "general_question", False, ["hinglish"]),
    ("mera account kaise delete kare pura data ke saath", "general_question", False, ["hinglish"]),
    ("aapne mera card galat charge kiya", "pricing", True, ["hinglish"]),
    ("mujhe weekly report chahiye kaise set kare", "product_question", False, ["hinglish"]),
    ("aapki app Hindi mein available hai kya", "product_question", False, ["hinglish"]),
    ("maine 3 baar email kiya koi reply nahi aaya", "complaint", True, ["hinglish"]),
    ("kya aap mere city mein service dete ho", "general_question", False, ["hinglish"]),
    ("mera account kaise recover kare", "account_issue", False, ["hinglish"]),
    ("mujhe aapke product ki training chahiye", "sales", False, ["hinglish"]),
    ("aapka refund process kaise kaam karta hai", "refund", False, ["hinglish"]),
    ("kya main apna billing cycle change kar sakta hu", "pricing", False, ["hinglish"]),
    ("mera email notifications nahi aa raha", "technical_support", False, ["hinglish"]),
    ("aapne mujhe block kyu kar diya unblock karo", "account_issue", True, ["hinglish"]),
    ("mujhe senior se baat karni hai urgent", "human_request", True, ["hinglish"]),
    ("aapki service se bahut disappointment hai", "complaint", False, ["hinglish"]),
    ("mujhe aapke software ke bare mein bataye", "general_question", False, ["hinglish"]),
    ("mera payment successful dikha raha hai but credits nahi aaye", "technical_support", True, ["hinglish"]),
    ("kya main apna plan downgrade kar sakta hu", "pricing", False, ["hinglish"]),
    ("aapke yaha kya kya features hain batao", "product_question", False, ["hinglish"]),
    ("mujhe bulk SMS feature hai kya", "product_question", False, ["hinglish"]),
    ("aapka onboarding process bahut complicated hai", "complaint", False, ["hinglish"]),
    ("mera kaam ruka hai do din se koi respond nahi kar raha", "complaint", True, ["hinglish"]),
    ("kya aapke yaha setup fee hai", "pricing", False, ["hinglish"]),
    ("mujhe live chat support chahiye", "general_question", False, ["hinglish"]),
    ("aapki app storage bahut le rahi hai phone mein", "technical_support", False, ["hinglish"]),
    ("mera invoice download kaise kare", "general_question", False, ["hinglish"]),
    ("aapne mera account suspend kar diya kyun", "account_issue", True, ["hinglish"]),
    ("mujhe custom domain chahiye kaise set kare", "product_question", False, ["hinglish"]),
    ("kya aap WhatsApp support dete ho", "general_question", False, ["hinglish"]),
    ("mera trial khatam hone wala hai kya karna hoga", "general_question", False, ["hinglish"]),
    ("aapki pricing transparent nahi hai", "pricing", False, ["hinglish"]),
    ("maine support ticket khola lekin koi assign nahi hua", "technical_support", True, ["hinglish"]),
    ("mujhe aapke platform se data migrate karna hai", "sales", False, ["hinglish"]),
    ("aapki app baar baar hang ho jati hai", "technical_support", False, ["hinglish"]),
    ("mera plan upgrade ka option kahan hai", "sales", False, ["hinglish"]),
    ("aapne mujhe double charge kiya hai immediately refund karo", "refund", True, ["hinglish"]),
    ("kya main apna workspace delete kar sakta hu", "general_question", False, ["hinglish"]),
    ("mujhe API access chahiye kaise le", "sales", False, ["hinglish"]),
    ("mera payment fail ho gaya but paisa kat liya", "technical_support", True, ["hinglish"]),
    ("aapki website par 500 error aa raha hai", "technical_support", False, ["hinglish"]),
    ("mujhe aapke product ki ek feature list chahiye", "product_question", False, ["hinglish"]),
    ("kya aapke yaha GST invoice milta hai", "pricing", False, ["hinglish"]),
    ("mera email change nahi ho raha help karo", "account_issue", False, ["hinglish"]),
    ("aapka mobile app Android pe kyun nahi chal raha", "technical_support", False, ["hinglish"]),
    ("mujhe enterprise plan ka demo chahiye", "sales", False, ["hinglish"]),
    ("aapki service band kar raha hu refund do warna legal action", "refund", True, ["hinglish"]),
    ("aapka product theek hai but support kharab hai", "complaint", False, ["hinglish"]),
    ("mujhe training chahiye product ki kya facility hai", "product_question", False, ["hinglish"]),
    ("mera password reset ka link bhejo", "account_issue", False, ["hinglish"]),
    ("kya aap credit card ke alawa bhi payment accept karte ho", "pricing", False, ["hinglish"]),
    ("maine galati se account delete kar diya recover karo", "account_issue", True, ["hinglish"]),
    ("aapki app ki quality improve karo warna switch kar dunga", "complaint", True, ["hinglish"]),
    ("mujhe product ke baare mein jaankari chahiye", "general_question", False, ["hinglish"]),
    ("aapne mera plan downgrade kyun kar diya", "account_issue", True, ["hinglish"]),
    ("mera invoice me galti hai correct karo", "pricing", False, ["hinglish"]),
    ("kya aapke yaha EMI option hai", "pricing", False, ["hinglish"]),
    ("mujhe aapki service pasand nahi aayi", "complaint", False, ["hinglish"]),
    ("maine aapko 4 din pehle email kiya tha reply kyun nahi aaya", "complaint", True, ["hinglish"]),
    ("kya aapke yaha COD available hai", "pricing", False, ["hinglish"]),
    ("mera account kaise reactivate kare", "account_issue", False, ["hinglish"]),
    ("aapka software mere system pe nahi chal raha", "technical_support", False, ["hinglish"]),
    ("mujhe bulk email feature chahiye kya aapke paas hai", "product_question", False, ["hinglish"]),
    ("main aapka product 2 saal se use kar raha hu update nahi aaya", "complaint", False, ["hinglish"]),
    ("aapne mujhe wrong currency mein bill kiya", "pricing", True, ["hinglish"]),
    ("mujhe koi experienced person se baat karni hai", "human_request", True, ["hinglish"]),
    ("mera data kab tak recover hoga", "technical_support", True, ["hinglish"]),
    ("aapke server down hain kya", "technical_support", True, ["hinglish"]),
    ("mujhe aapke software ki trial chahiye", "sales", False, ["hinglish"]),
    ("kya mein apna account temporarily suspend kar sakta hu", "general_question", False, ["hinglish"]),
    ("meri team ke liye alag workspace chahiye", "sales", False, ["hinglish"]),
    ("aapka pricing page clear nahi hai", "pricing", False, ["hinglish"]),
    ("mujhe aapke software ke through invoice generate karna hai", "general_question", False, ["hinglish"]),
    ("aapki app ne mera data corrupt kar diya", "technical_support", True, ["hinglish"]),
    ("mera refund kab tak aayega", "refund", True, ["hinglish"]),
    ("aapke yaha kaunsa payment gateway use hota hai", "general_question", False, ["hinglish"]),
    ("mujhe aapke product ki video tutorial chahiye", "general_question", False, ["hinglish"]),
    ("mera email spam folder mein ja raha hai", "technical_support", False, ["hinglish"]),
    ("aapne mujhe overcharge kiya hai check karo", "pricing", True, ["hinglish"]),
    ("mujhe custom report banana hai kaise kare", "product_question", False, ["hinglish"]),
    ("aapka software bahut complicated hai easy banao", "complaint", False, ["hinglish"]),
    ("maine 2 bar pay kar diya ek refund karo", "refund", True, ["hinglish"]),
    ("kya main apna number change kar sakta hu account mein", "account_issue", False, ["hinglish"]),
    ("aapki team ne mera issue resolve kiya thanks", "technical_support", False, ["hinglish", "resolution_state"]),
    ("mujhe aapke yaha job chahiye", "other", False, ["hinglish"]),
    ("aapki app ko Hindi support add karo", "product_question", False, ["hinglish"]),
    ("mera billing cycle change karo monthly se yearly", "pricing", False, ["hinglish"]),
    ("aapne mujhe refund ka confirmation nahi bheja", "refund", False, ["hinglish"]),
    ("kya aapke yaha volume discount available hai 50 users ke liye", "sales", False, ["hinglish"]),
    ("mujhe aapki service ka SLA document chahiye", "general_question", False, ["hinglish"]),
    ("aapka product use karke bahut time ho gaya lekin update nahi aaya", "complaint", False, ["hinglish"]),
    ("mujhe aapke integration ke bare mein bataye Salesforce ke saath", "product_question", False, ["hinglish"]),
    ("mera payment method update nahi ho raha", "account_issue", False, ["hinglish"]),
    ("aapne mujhe wrong features activate kiye hain", "technical_support", False, ["hinglish"]),
    ("mujhe aapki service recommend karni hai kisi ko kaise kare", "general_question", False, ["hinglish"]),
    ("aapka mobile app bahut heavy hai phone hang ho jata hai", "technical_support", False, ["hinglish"]),
    ("mera API key regenerate karna hai", "general_question", False, ["hinglish"]),
    ("kya aapke yaha koi community forum hai", "general_question", False, ["hinglish"]),
    ("mujhe aapki service se koi help nahi mili", "complaint", False, ["hinglish"]),
    ("aapki app ke features kya hain", "product_question", False, ["hinglish"]),
    ("mera account ka status check karo", "account_issue", False, ["hinglish"]),
    ("mujhe aapke yaha bulk messaging feature chahiye", "product_question", False, ["hinglish"]),
    ("aapki team bahut slow hai kaam karo", "complaint", True, ["hinglish"]),
    ("maine aapko 10 baar phone kiya koi nahi utha", "human_request", True, ["hinglish"]),
    ("mera trial expire hone wala hai extend karo", "sales", False, ["hinglish"]),
    ("aapne mujhe wrong price quote diya tha", "pricing", False, ["hinglish"]),
    ("kya main aapke platform par multiple accounts bana sakta hu", "general_question", False, ["hinglish"]),
    ("mujhe aapki service ke baare mein online reviews nahi mile", "general_question", False, ["hinglish"]),
    ("aapki app mere phone mein notifications nahi dikha rahi", "technical_support", False, ["hinglish"]),
    ("meri report mein data missing hai", "technical_support", False, ["hinglish"]),
    ("aapne mujhe galat features bataye the sales team ne", "complaint", False, ["hinglish"]),
    ("mujhe aapki service ka test drive lena hai", "sales", False, ["hinglish"]),
    ("aapki app ki design outdated hai", "complaint", False, ["hinglish"]),
    ("mera account mein kisi aur ka data aa raha hai", "account_issue", True, ["hinglish"]),
    ("mujhe aapke yaha payroll integration chahiye", "product_question", False, ["hinglish"]),
    ("aapne mera auto payment ka option enable kar diya band karo", "pricing", False, ["hinglish"]),
    ("mera API limit exceed ho gaya hai increase karo", "sales", False, ["hinglish"]),
    ("aapki service ki quality pehle acchi thi ab nahi", "complaint", False, ["hinglish"]),
]

for item in hinglish_intents:
    add(*item)

# ============================================================
# 5. NOISY/REAL-WORLD (+300 new)
# ============================================================
batch_add([
    ("plz hlp my accnt is locked", "account_issue", True, ["noisy"], "v3 noisy"),
    ("refund my mony r i repot u", "refund", True, ["noisy"], "v3 noisy"),
    ("need sm1 2 call me urgent", "human_request", True, ["noisy"], "v3 noisy"),
    ("how much 4 premium plaN", "pricing", False, ["noisy"], "v3 noisy"),
    ("i cnt login since ystrday hlp", "account_issue", True, ["noisy"], "v3 noisy"),
    ("y is this app so slo", "technical_support", False, ["noisy"], "v3 noisy"),
    ("my paymnt got dedctd twice", "refund", True, ["noisy"], "v3 noisy"),
    ("wher can i see my invoce", "pricing", False, ["noisy"], "v3 noisy"),
    ("i want 2 canc3l my plan", "sales", False, ["noisy"], "v3 noisy"),
    ("ur chargin me wrng ammount", "pricing", True, ["noisy"], "v3 noisy"),
    ("pls refund my mony i need it urgent", "refund", True, ["noisy"], "v3 noisy"),
    ("i dnt no how 2 use this app", "general_question", False, ["noisy"], "v3 noisy"),
    ("can u help me pls im stuck", "general_question", False, ["noisy"], "v3 noisy"),
    ("my dashbord is not loadin", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i want a humen not a bot", "human_request", True, ["noisy"], "v3 noisy"),
    ("ur servis is a scam i want my mony bak", "complaint", True, ["noisy"], "v3 noisy"),
    ("hw do i upgrade my plan", "sales", False, ["noisy"], "v3 noisy"),
    ("the buton is not workin on my page", "technical_support", False, ["noisy"], "v3 noisy"),
    ("my accnt got hakd help me plz", "account_issue", True, ["noisy"], "v3 noisy"),
    ("y r u chargin me so much", "pricing", False, ["noisy"], "v3 noisy"),
    ("i cnt chang my email adres", "account_issue", False, ["noisy"], "v3 noisy"),
    ("the app is frezzing evry 5 min", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i wana cancel my subskripshun", "sales", False, ["noisy"], "v3 noisy"),
    ("u guyz chargd me rong amnt", "pricing", True, ["noisy"], "v3 noisy"),
    ("plese refund my muney", "refund", True, ["noisy"], "v3 noisy"),
    ("i dnt understand the pricin", "pricing", False, ["noisy"], "v3 noisy"),
    ("my account got haxed", "account_issue", True, ["noisy"], "v3 noisy"),
    ("cn u help me with loggin in", "account_issue", False, ["noisy"], "v3 noisy"),
    ("i paid but not gettin acsess", "account_issue", True, ["noisy"], "v3 noisy"),
    ("this servic is a scam", "complaint", True, ["noisy"], "v3 noisy"),
    ("i wnt my mony bak", "refund", True, ["noisy"], "v3 noisy"),
    ("hw do i delet my acc", "general_question", False, ["noisy"], "v3 noisy"),
    ("the dashbord is showin rong data", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i cnt chanj my email", "account_issue", False, ["noisy"], "v3 noisy"),
    ("my sso is brokn", "technical_support", False, ["noisy"], "v3 noisy"),
    ("the app crahses evrytime i opn", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i m gettin eror code 500", "technical_support", False, ["noisy"], "v3 noisy"),
    ("cn u call me pls", "human_request", True, ["noisy"], "v3 noisy"),
    ("wher is the upgarde buton", "product_question", False, ["noisy"], "v3 noisy"),
    ("my notifcations arnt workin", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i payd for premum but dnt hav it", "account_issue", True, ["noisy"], "v3 noisy"),
    ("hw to add team membrs", "general_question", False, ["noisy"], "v3 noisy"),
    ("the srch bar is not wrking", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i wnt a humen to hlp me", "human_request", True, ["noisy"], "v3 noisy"),
    ("my invoic is rong", "pricing", False, ["noisy"], "v3 noisy"),
    ("2 factor auth not wrking", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i cnt c my billng info", "pricing", False, ["noisy"], "v3 noisy"),
    ("the export buttn does nthng", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i need a refnd 4 last mnth", "refund", True, ["noisy"], "v3 noisy"),
    ("ur support is helpless", "complaint", False, ["noisy"], "v3 noisy"),
    ("this sftware is usless", "complaint", False, ["noisy"], "v3 noisy"),
    ("hw 2 get a demo", "sales", False, ["noisy"], "v3 noisy"),
    ("cnt login to my acc", "account_issue", False, ["noisy"], "v3 noisy"),
    ("my paymnt was deductd twice", "refund", True, ["noisy"], "v3 noisy"),
    ("i am not able to acces my acc", "account_issue", False, ["noisy"], "v3 noisy"),
    ("plez help im lockt out", "account_issue", True, ["noisy"], "v3 noisy"),
    ("the websit is not loadng", "technical_support", False, ["noisy"], "v3 noisy"),
    ("my acount was suspndd", "account_issue", True, ["noisy"], "v3 noisy"),
    ("i wnt my subscripion canclld", "sales", False, ["noisy"], "v3 noisy"),
    ("hw do i get in tuch with sals", "sales", False, ["noisy"], "v3 noisy"),
    ("i forgat my pasword", "account_issue", False, ["noisy"], "v3 noisy"),
    ("the app is to slow", "technical_support", False, ["noisy"], "v3 noisy"),
    ("plz fix my accnt issue", "account_issue", False, ["noisy"], "v3 noisy"),
    ("i hav been wating 4 2 weks", "complaint", True, ["noisy"], "v3 noisy"),
    ("my id is not verifide", "account_issue", False, ["noisy"], "v3 noisy"),
    ("can smeone hlp me plzzz", "general_question", False, ["noisy"], "v3 noisy"),
    ("i m very upset with ur servise", "complaint", False, ["noisy"], "v3 noisy"),
    ("refnd my mony or i su", "refund", True, ["noisy"], "v3 noisy"),
    ("i cnt updat my profil", "account_issue", False, ["noisy"], "v3 noisy"),
    ("hw many users can i ad", "product_question", False, ["noisy"], "v3 noisy"),
    ("my workflo is not runing", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i wnt 2 spek to a manager", "human_request", True, ["noisy"], "v3 noisy"),
    ("the dashbord is 2 complicatd", "complaint", False, ["noisy"], "v3 noisy"),
    ("i dnt kno hw 2 use this app", "general_question", False, ["noisy"], "v3 noisy"),
    ("my teammates cnt join the workspac", "account_issue", False, ["noisy"], "v3 noisy"),
    ("the notifcatn icon is not showng", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i m gettin erors in my report", "technical_support", False, ["noisy"], "v3 noisy"),
    ("the api integratin is brokn", "technical_support", False, ["noisy"], "v3 noisy"),
    ("my acnt is lokcd help", "account_issue", True, ["noisy"], "v3 noisy"),
    ("i ws chargd the rong plan", "pricing", True, ["noisy"], "v3 noisy"),
    ("hw to updgrade my acccount", "sales", False, ["noisy"], "v3 noisy"),
    ("the chat bot is not helpfl", "complaint", False, ["noisy"], "v3 noisy"),
    ("i wnt a humn not a bot plzzz", "human_request", True, ["noisy"], "v3 noisy"),
    ("my data is not syncing propely", "technical_support", False, ["noisy"], "v3 noisy"),
    ("whr can i find the pricin page", "pricing", False, ["noisy"], "v3 noisy"),
    ("i neeed help with API integratn", "technical_support", False, ["noisy"], "v3 noisy"),
    ("u guyz charge too much", "pricing", False, ["noisy"], "v3 noisy"),
    ("my api key brok", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i dont lik this servis at all", "complaint", False, ["noisy"], "v3 noisy"),
    ("hlo i need ur help regarding billing", "pricing", False, ["noisy"], "v3 noisy"),
    ("this iz not wot i paid 4", "complaint", False, ["noisy"], "v3 noisy"),
    ("i wnt 2 speek to a reel person", "human_request", True, ["noisy"], "v3 noisy"),
    ("refnd my payment or i will su u", "refund", True, ["noisy"], "v3 noisy"),
    ("hw can i upgrade my plan", "sales", False, ["noisy"], "v3 noisy"),
    ("the buton is not wrking", "technical_support", False, ["noisy"], "v3 noisy"),
    ("my team membr cnt join the workspac", "account_issue", False, ["noisy"], "v3 noisy"),
    ("i accidently deletd my accunt", "account_issue", True, ["noisy"], "v3 noisy"),
    ("wer r my reports", "technical_support", False, ["noisy"], "v3 noisy"),
    ("i cnt find the settngs buton", "technical_support", False, ["noisy"], "v3 noisy"),
    ("my pasword resets isnt workng", "account_issue", False, ["noisy"], "v3 noisy"),
    ("the app is frezzing everytim i open", "technical_support", False, ["noisy"], "v3 noisy"),
    ("plese fix my paymnt issue", "pricing", False, ["noisy"], "v3 noisy"),
    ("i cnt figur out hw 2 reset passwrd", "account_issue", False, ["noisy"], "v3 noisy"),
    ("wht is the price of premium package", "pricing", False, ["noisy"], "v3 noisy"),
])

# ============================================================
# 6. MULTI-INTENT (~250 new)
# ============================================================
batch_add([
    ("Can you fix my account? Also how much for premium?", "account_issue", False, ["multi_intent"], "v3 multi-intent"),
    ("I'm furious about my billing and I want a refund", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("Tell me about your product and your pricing", "product_question", False, ["multi_intent"], "v3 multi-intent"),
    ("I want to buy but first I need help with my login", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Technical support needed but also tell me about pricing", "technical_support", False, ["multi_intent"], "v3 multi-intent"),
    ("Your service is terrible and I want to speak to a manager", "human_request", True, ["multi_intent"], "v3 multi-intent"),
    ("I need a human to help me purchase", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Payment failed and I want my money back", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("I want to cancel my account and get a refund", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("My account got hacked and I need to speak to someone", "account_issue", True, ["multi_intent"], "v3 multi-intent"),
    ("I'm interested in your product but your pricing seems high", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("Can you help me reset my password and also tell me about plans", "general_question", False, ["multi_intent"], "v3 multi-intent"),
    ("I want to complain about your service and get a refund", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("Need technical help with integration and also want a price quote", "technical_support", False, ["multi_intent"], "v3 multi-intent"),
    ("I'm locked out of my account and I'm really angry about it", "account_issue", True, ["multi_intent"], "v3 multi-intent"),
    ("I want to buy but your site keeps crashing during checkout", "sales", True, ["multi_intent"], "v3 multi-intent"),
    ("Mujhe refund chahiye aur account bhi fix karo", "refund", True, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("Main to sirf pricing poochh raha tha but ab account bhi nahi chal raha", "pricing", False, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("I need both technical support and a refund for the downtime", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("I'm interested in buying but I need to understand the features first", "product_question", False, ["multi_intent"], "v3 multi-intent"),
    ("I was going to complain but now I just want my money back", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("I have an account issue and it's making me very angry", "account_issue", False, ["multi_intent"], "v3 multi-intent"),
    ("Your pricing page is confusing and I want to speak to sales", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Mera account issue hai aur refund bhi chahiye", "refund", True, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("I have a complaint about billing but I need technical help accessing my account", "account_issue", True, ["multi_intent"], "v3 multi-intent"),
    ("Pricing to theek hai but product features ke baare mein batao", "product_question", False, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("I want to buy but first help me understand the pricing", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("Cancel my subscription and tell me about your other products", "refund", False, ["multi_intent"], "v3 multi-intent"),
    ("I'm locked out of my account and I want compensation", "account_issue", True, ["multi_intent"], "v3 multi-intent"),
    ("My payment failed and I need help from a human", "technical_support", True, ["multi_intent"], "v3 multi-intent"),
    ("I want to upgrade my plan but I'm confused about the pricing", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("Your software has a bug and I'm not happy about the price either", "technical_support", False, ["multi_intent"], "v3 multi-intent"),
    ("I need a human to explain the features and pricing to me", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Complaint about your service and question about features", "complaint", False, ["multi_intent"], "v3 multi-intent"),
    ("Account recovery needed and I want to talk to a manager", "account_issue", True, ["multi_intent"], "v3 multi-intent"),
    ("I have both a sales inquiry and a technical problem", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Refund request and also need help with account closure", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("Can I speak to technical support about a billing issue", "technical_support", False, ["multi_intent"], "v3 multi-intent"),
    ("I need help with pricing and my account is also locked", "pricing", True, ["multi_intent"], "v3 multi-intent"),
    ("This is both a feature request and a complaint about existing features", "complaint", False, ["multi_intent"], "v3 multi-intent"),
    ("I want to buy the plan but I need technical help with setup first", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Sales team se baat karni hai aur price bhi confirm karna hai", "sales", False, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("I'm furious. Someone hacked my account. Call me now.", "account_issue", True, ["multi_intent"], "v3 multi-intent"),
    ("Not asking for a refund but your service is terrible", "complaint", False, ["multi_intent"], "v3 multi-intent"),
    ("I don't need a human, just tell me how to fix this error", "technical_support", False, ["multi_intent"], "v3 multi-intent"),
    ("I'll take legal action if I don't get my money back", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("Can I speak to sales? Actually, just tell me the price.", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("Human request hai but also my payment failed", "human_request", True, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("Your chatbot is useless but I don't want to escalate", "complaint", False, ["multi_intent"], "v3 multi-intent"),
    ("Let me talk to a manager. This is the third time I'm calling.", "human_request", True, ["multi_intent"], "v3 multi-intent"),
    ("I was angry earlier but the issue has been resolved", "complaint", False, ["multi_intent", "resolution_state"], "v3 multi-intent"),
    ("My payment failed yesterday but it's working now", "technical_support", False, ["multi_intent", "resolution_state"], "v3 multi-intent"),
    ("I don't want a refund anymore. Cancel my request.", "refund", False, ["multi_intent", "negation"], "v3 multi-intent"),
    ("Can you explain the refund policy and also your pricing", "general_question", False, ["multi_intent"], "v3 multi-intent"),
    ("Product question: does this integrate with Slack? Also how much?", "product_question", False, ["multi_intent"], "v3 multi-intent"),
    ("Technical support needed but I want a human on the line", "technical_support", True, ["multi_intent"], "v3 multi-intent"),
    ("I have a billing issue but it's not urgent. Just explain.", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("I want to cancel my subscription but first explain the refund policy", "refund", False, ["multi_intent"], "v3 multi-intent"),
    ("I'm not sure if I need a refund or just an explanation", "refund", False, ["multi_intent"], "v3 multi-intent"),
    ("This is both a complaint and a question about pricing", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("I have sales inquiry but also a technical problem", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Can I speak to technical support? Actually just tell me the price.", "pricing", False, ["multi_intent"], "v3 multi-intent"),
    ("I want to talk to a human about my refund", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("Aapki service achi hai lekin price bahut zyada hai", "pricing", False, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("Payment failed and customer support nahi mil raha. Help.", "technical_support", True, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
    ("I need both technical support and a refund", "refund", True, ["multi_intent"], "v3 multi-intent"),
    ("I'm interested in buying but your site keeps crashing", "sales", False, ["multi_intent"], "v3 multi-intent"),
    ("Mujhe refund aur technical support dono chahiye", "refund", True, ["multi_intent", "hinglish"], "v3 multi-intent hinglish"),
])

# ============================================================
# 7. INFORMATION VS ACTION (~100 new)
# ============================================================
batch_add([
    ("What is your refund policy?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Process my refund now.", "refund", True, ["standard"], "v3 info vs action"),
    ("Can you explain how account cancellation works?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Cancel my account now.", "sales", False, ["standard"], "v3 info vs action"),
    ("What happens if I cancel?", "general_question", False, ["standard"], "v3 info vs action"),
    ("I want to cancel. Process it.", "sales", False, ["standard"], "v3 info vs action"),
    ("How do I get a refund?", "refund", False, ["standard"], "v3 info vs action"),
    ("Give me my money back immediately.", "refund", True, ["standard"], "v3 info vs action"),
    ("What is the process for upgrading?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Upgrade my plan right now.", "sales", False, ["standard"], "v3 info vs action"),
    ("Can you tell me about your pricing tiers?", "pricing", False, ["standard"], "v3 info vs action"),
    ("Sign me up for the premium tier now.", "sales", False, ["standard"], "v3 info vs action"),
    ("How does the auto-renewal work?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Turn off my auto-renewal immediately.", "sales", False, ["standard"], "v3 info vs action"),
    ("What do I do if I forget my password?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Reset my password now. I can't wait.", "account_issue", True, ["standard"], "v3 info vs action"),
    ("How do I contact your support team?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Connect me to your support team now.", "human_request", True, ["standard"], "v3 info vs action"),
    ("What is the SLA for resolving issues?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Fix this issue within the SLA timeframe.", "technical_support", True, ["standard"], "v3 info vs action"),
    ("Can you explain the difference between plans?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Switch me to the enterprise plan immediately.", "sales", False, ["standard"], "v3 info vs action"),
    ("Tell me more about security features.", "product_question", False, ["standard"], "v3 info vs action"),
    ("Enable two-factor authentication on my account now.", "account_issue", False, ["standard"], "v3 info vs action"),
    ("What causes the 500 error?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Fix the 500 error on my dashboard now.", "technical_support", True, ["standard"], "v3 info vs action"),
    ("How do I export my data?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Export all my data immediately.", "general_question", False, ["standard"], "v3 info vs action"),
    ("What is your data retention policy?", "general_question", False, ["standard"], "v3 info vs action"),
    ("Delete all my data from your servers now.", "general_question", True, ["standard"], "v3 info vs action"),
])

# ============================================================
# 8. NEGATION (~150 new)
# ============================================================
batch_add([
    ("Don't refund me, just explain the policy", "general_question", False, ["negation"], "v3 negation"),
    ("Don't transfer me, just help me here", "general_question", False, ["negation"], "v3 negation"),
    ("I don't need a human, just tell me how to fix this", "technical_support", False, ["negation"], "v3 negation"),
    ("I'm not asking for a refund, I just want to complain", "complaint", False, ["negation"], "v3 negation"),
    ("The problem is no longer happening", "technical_support", False, ["negation", "resolution_state"], "v3 negation"),
    ("I don't want my money back. I just want answers.", "general_question", False, ["negation"], "v3 negation"),
    ("Not looking for a refund. Just want to understand the charge.", "pricing", False, ["negation"], "v3 negation"),
    ("I do not want to speak to anyone. Just answer here.", "general_question", False, ["negation"], "v3 negation"),
    ("Don't process any refund. I was mistaken about the charge.", "pricing", False, ["negation"], "v3 negation"),
    ("No I don't need a call. Email is sufficient.", "general_question", False, ["negation"], "v3 negation"),
    ("I am NOT requesting escalation. Just feedback.", "complaint", False, ["negation"], "v3 negation"),
    ("Please don't cancel my account. I want to keep it.", "general_question", False, ["negation"], "v3 negation"),
    ("I don't want compensation. I just want the issue fixed.", "technical_support", False, ["negation"], "v3 negation"),
    ("Do not escalate this ticket. It's a simple question.", "general_question", False, ["negation"], "v3 negation"),
    ("I'm not angry. I just want to report something.", "general_question", False, ["negation"], "v3 negation"),
    ("Nobody needs to follow up with me. I'm all set.", "general_question", False, ["negation"], "v3 negation"),
    ("I don't want to buy anything. Just checking info.", "general_question", False, ["negation"], "v3 negation"),
    ("This is not a support request. It's a suggestion.", "other", False, ["negation"], "v3 negation"),
    ("Don't worry about this. I'll handle it.", "general_question", False, ["negation"], "v3 negation"),
    ("I don't need help anymore. Figured it out.", "technical_support", False, ["negation", "resolution_state"], "v3 negation"),
    ("Not a complaint. Just an observation.", "other", False, ["negation"], "v3 negation"),
    ("No urgent action needed. Take your time.", "general_question", False, ["negation"], "v3 negation"),
    ("I expressly forbid any escalation of this matter.", "general_question", False, ["negation"], "v3 negation"),
    ("Do not contact me about this issue. Ever.", "general_question", False, ["negation"], "v3 negation"),
    ("I don't want to talk to a manager. You're fine.", "general_question", False, ["negation"], "v3 negation"),
    ("Not interested in purchasing right now.", "pricing", False, ["negation"], "v3 negation"),
    ("I do NOT consent to being transferred.", "general_question", False, ["negation"], "v3 negation"),
    ("Please don't call me for feedback or anything else.", "general_question", False, ["negation"], "v3 negation"),
    ("I'm not disputing the charge. Just asking about it.", "pricing", False, ["negation"], "v3 negation"),
    ("No refund needed. Just wanted to provide feedback.", "complaint", False, ["negation"], "v3 negation"),
    ("Don't send me to billing. I don't need that.", "general_question", False, ["negation"], "v3 negation"),
    ("I'm not asking for human intervention. You answered it.", "general_question", False, ["negation"], "v3 negation"),
    ("Don't bother your support team. I solved it.", "technical_support", False, ["negation", "resolution_state"], "v3 negation"),
    ("I will not accept a call. Text only.", "general_question", False, ["negation"], "v3 negation"),
    ("This does not require immediate attention.", "general_question", False, ["negation"], "v3 negation"),
    ("I don't want a refund. I want the service to work.", "technical_support", True, ["negation"], "v3 negation"),
    ("Refund mat karo. Bas mujhe batao kya hua.", "pricing", False, ["negation", "hinglish"], "v3 negation hinglish"),
    ("Mujhe human se baat nahi karni. Bas jawab do.", "general_question", False, ["negation", "hinglish"], "v3 negation hinglish"),
    ("Maine refund nahi manga. Sirf puchh raha hu.", "general_question", False, ["negation", "hinglish"], "v3 negation hinglish"),
    ("Koi escalate mat karo. Simple sawaal hai.", "general_question", False, ["negation", "hinglish"], "v3 negation hinglish"),
    ("Mujhe koi call nahi chahiye. Bas email karo.", "general_question", False, ["negation", "hinglish"], "v3 negation hinglish"),
])

# ============================================================
# 9. RESOLUTION STATE (~100 new)
# ============================================================
batch_add([
    ("My account was locked but support fixed it", "account_issue", False, ["resolution_state"], "v3 resolution"),
    ("I was angry about the billing error but it's corrected now", "complaint", False, ["resolution_state"], "v3 resolution"),
    ("The bug I reported last week has been fixed. Thanks.", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("My refund was processed. Thank you for your help.", "refund", False, ["resolution_state"], "v3 resolution"),
    ("Everything is working now. You can close my ticket.", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("The issue I had with login is resolved. Thanks!", "account_issue", False, ["resolution_state"], "v3 resolution"),
    ("I was frustrated earlier but your team fixed everything.", "complaint", False, ["resolution_state"], "v3 resolution"),
    ("The error I reported is gone after the patch. Thanks!", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("My account is back to normal. Appreciate the help.", "account_issue", False, ["resolution_state"], "v3 resolution"),
    ("The double charge was refunded. Issue closed.", "refund", False, ["resolution_state"], "v3 resolution"),
    ("After the update, everything started working fine.", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("I complained earlier but the matter is resolved now.", "complaint", False, ["resolution_state"], "v3 resolution"),
    ("My subscription was correctly set up after your help.", "sales", False, ["resolution_state"], "v3 resolution"),
    ("The missing data was restored. Thank you!", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("I was locked out but your team fixed it promptly.", "account_issue", False, ["resolution_state"], "v3 resolution"),
    ("The migration issue is resolved. Works perfectly.", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("My refund request was processed in time. Thank you.", "refund", False, ["resolution_state"], "v3 resolution"),
    ("Earlier I had a pricing question but it's clear now.", "pricing", False, ["resolution_state"], "v3 resolution"),
    ("The bug that crashed my reports is fixed. Thanks team.", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("I don't need further assistance. Issue is resolved.", "technical_support", False, ["resolution_state"], "v3 resolution"),
    ("Mera issue solve ho gaya. Ticket close karo.", "technical_support", False, ["resolution_state", "hinglish"], "v3 resolution hinglish"),
    ("Maine jo problem batayi thi wo fix ho gayi. Thanks!", "technical_support", False, ["resolution_state", "hinglish"], "v3 resolution hinglish"),
    ("Mera refund aa gaya. Sab theek hai.", "refund", False, ["resolution_state", "hinglish"], "v3 resolution hinglish"),
    ("Payment issue resolved ho gaya. Ab sab sahi hai.", "technical_support", False, ["resolution_state", "hinglish"], "v3 resolution hinglish"),
    ("Pehle account lock tha but ab theek hai.", "account_issue", False, ["resolution_state", "hinglish"], "v3 resolution hinglish"),
])

# ============================================================
# BALANCE: Add more examples for underrepresented intents
# Check current distribution then add targeted examples
# ============================================================

# Additional general_question, product_question, pricing, sales, other
batch_add([
    # general_question
    ("Is there a limit on how many projects I can create?", "general_question", False, ["standard"], "v3 balance"),
    ("Can I collaborate with external partners on your platform?", "general_question", False, ["standard"], "v3 balance"),
    ("Do you have a roadmap of upcoming features?", "general_question", False, ["standard"], "v3 balance"),
    ("How do I set permissions for each team member?", "general_question", False, ["standard"], "v3 balance"),
    ("Can I use your API to build a custom integration?", "product_question", False, ["standard"], "v3 balance"),
    ("Does your platform support webhooks for real-time updates?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I schedule automated email campaigns with your tool?", "product_question", False, ["standard"], "v3 balance"),
    ("Does the mobile app support offline mode?", "product_question", False, ["standard"], "v3 balance"),
    # pricing
    ("What is the cost per user for the enterprise tier?", "pricing", False, ["standard"], "v3 balance"),
    ("Is VAT included in your pricing?", "pricing", False, ["standard"], "v3 balance"),
    ("Do you charge for data storage separately?", "pricing", False, ["standard"], "v3 balance"),
    ("What is the pricing for API-only access?", "pricing", False, ["standard"], "v3 balance"),
    # sales
    ("Please set up an account for me on the premium plan.", "sales", False, ["standard"], "v3 balance"),
    ("I need to purchase additional seats for my team.", "sales", False, ["standard"], "v3 balance"),
    ("I want to talk to someone about a custom enterprise deal.", "sales", False, ["standard"], "v3 balance"),
    ("Please process my upgrade to the annual plan.", "sales", False, ["standard"], "v3 balance"),
    # other
    ("Have a wonderful day!", "other", False, ["standard"], "v3 balance"),
    ("Thanks for resolving my issue quickly.", "other", False, ["standard"], "v3 balance"),
    ("I appreciate your support team's effort.", "other", False, ["standard"], "v3 balance"),
    ("You guys are doing a great job.", "other", False, ["standard"], "v3 balance"),
    # technical_support additional
    ("The recurring task automation stopped triggering.", "technical_support", False, ["standard"], "v3 balance"),
    ("Email templates are not rendering correctly in Outlook.", "technical_support", False, ["standard"], "v3 balance"),
    ("The bulk edit feature is not applying changes to all selected items.", "technical_support", False, ["standard"], "v3 balance"),
    ("My dashboard filters keep resetting after page refresh.", "technical_support", False, ["standard"], "v3 balance"),
    # account_issue additional
    ("I need to update my business address on the account.", "account_issue", False, ["standard"], "v3 balance"),
    ("My secondary email is not receiving verification code.", "account_issue", False, ["standard"], "v3 balance"),
    # human_request additional
    ("Please let me speak to someone who handles complex issues.", "human_request", True, ["standard"], "v3 balance"),
    ("I need to talk to a person about my account security.", "human_request", True, ["standard"], "v3 balance"),
    # refund - some information seeking vs demanding
    ("How long do refunds usually take to process?", "refund", False, ["standard"], "v3 balance"),
    ("What is the timeline for refund processing?", "refund", False, ["standard"], "v3 balance"),
    ("Refund my payment as per your 30-day policy.", "refund", True, ["standard"], "v3 balance"),
    ("I demand a refund for services not rendered.", "refund", True, ["standard"], "v3 balance"),
    # complaint
    ("Your product does not live up to the marketing claims.", "complaint", False, ["standard"], "v3 balance"),
    ("I feel misled by your sales team's promises.", "complaint", False, ["standard"], "v3 balance"),
])

# ============================================================
# BULK TEMPLATE GENERATION (~2000 additional)
# ============================================================

# general_question templates
gq_templates = [
    "How does {feature} work in your platform?",
    "Can you explain {feature}?",
    "What do I need to know about {feature}?",
    "Where can I find documentation about {feature}?",
    "Is there a tutorial for {feature}?",
    "What are the benefits of {feature}?",
    "How long does it take to set up {feature}?",
]
gq_features = [
    "two-factor authentication", "the dashboard", "email notifications",
    "data export", "the API", "role management", "audit logging",
    "the reporting module", "user permissions", "real-time sync",
    "backup", "the integration wizard", "custom fields",
    "the template system", "the import tool", "multi-factor auth",
    "the calendar sync", "the notification center", "billing",
    "the user directory", "content scheduling", "file sharing",
    "team management", "the mobile app", "performance monitoring",
    "the analytics dashboard", "the workflow builder",
]
for t in gq_templates:
    for f in gq_features:
        add(t.format(feature=f), "general_question", False, ["standard"], "v3 bulk gq")
        if len(examples) - v1_count > 4800:
            break
    if len(examples) - v1_count > 4800:
        break

# pricing templates
p_templates = [
    "How much for {feature}?",
    "What is the cost of {feature}?",
    "Is {feature} included in the base price?",
    "Do I pay extra for {feature}?",
]
p_features = [
    "additional users", "extra storage", "premium support",
    "API access", "the mobile app", "custom reporting",
    "white labeling", "SSO integration", "dedicated server",
    "the enterprise add-on", "migration", "training",
    "priority support", "data backup", "advanced analytics",
]
for t in p_templates:
    for f in p_features:
        add(t.format(feature=f), "pricing", False, ["standard", "pricing"], "v3 bulk pricing")
        if len(examples) - v1_count > 4900:
            break
    if len(examples) - v1_count > 4900:
        break

# tech support templates
ts_templates = [
    "{feature} is not working.",
    "{feature} keeps failing.",
    "I'm having trouble with {feature}.",
    "{feature} is broken after the latest update.",
    "Can you help me fix {feature}?",
    "I get an error when using {feature}.",
    "{feature} is showing incorrect data.",
    "{feature} is not loading properly.",
]
ts_features = [
    "The login page", "File upload", "Data export", "The search bar",
    "Email delivery", "The calendar sync", "The reporting dashboard",
    "The CSV importer", "My workflow automation", "The notification system",
    "The mobile app", "The API endpoint", "The webhook integration",
    "The drag and drop editor", "The template preview", "The backup system",
    "The bulk editor", "The chart generator", "The PDF export",
]
for t in ts_templates:
    for f in ts_features:
        add(t.format(feature=f), "technical_support", False, ["standard", "technical"], "v3 bulk tech")
        if len(examples) - v1_count > 5000:
            break
    if len(examples) - v1_count > 5000:
        break

# More Hinglish variations (template-free, natural)
batch_add([
    # Additional Hinglish
    ("aapka software setup kaise kare", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera account ka status check karo", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe premium plan ke features batao", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki service ka SLA kya hai", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera payment method verify nahi ho raha", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapke yaha custom integration kitne ka hai", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe apna account ka invoice chahiye", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki app ka dark mode kab aayega", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera trial period extend karo", "sales", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mujhe wrong currency charge kiya", "pricing", True, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe team ke saath workspace share karna hai", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapka API documentation kahan milega", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera auto payment band karo", "sales", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki service bahut achi hai lekin price zyada", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapki company ke baare mein batao", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mujhe galat plan assign kiya", "account_issue", True, ["hinglish"], "v3 bulk hinglish"),
    ("mera invoice kab generate hoga", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki app ko Hindi language support do", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapki service ka demo chahiye", "sales", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera account ka data export karo", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mera account block kyun kiya", "account_issue", True, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe bulk payment ka option chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki service ki quality improve karo", "complaint", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera password reset link kab tak aayega", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapke yaha annual plan pe kya discount hai", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke software ke latest update chahiye", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki app ne mera data delete kar diya", "technical_support", True, ["hinglish"], "v3 bulk hinglish"),
    ("mera account recover karne mein help karo", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapke yaha free trial kitne din ka hai", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke software ke through report generate karni hai", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapka customer support number kahan hai", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera payment gateway set up nahi ho raha", "technical_support", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki team ka response time bahut slow hai improve karo", "complaint", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke yaha multiple workspaces chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera API access enable karo", "sales", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mujhe double payment ka invoice bheja", "pricing", True, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke software ke source code access chahiye", "other", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera account ka billing history kaise dekhe", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki app baar baar crash ho rahi hai Android pe", "technical_support", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke yaha job apply karni hai", "other", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapke software mein Hindi font support hai kya", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera team member account add nahi ho raha", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mera annual plan ka amount galat liya", "pricing", True, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapki service ka refund chahiye service achhi nahi hai", "refund", True, ["hinglish", "multi_intent"], "v3 bulk hinglish"),
    ("aapke yaha bulk user upload ka feature hai", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera mobile number update karo account mein", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki app mein bahut bugs hain fix karo", "complaint", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapki service ka live demo chahiye", "sales", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera payment ka transaction ID check karo", "technical_support", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mujhe refund ka confirmation bheja tha but aaya nahi", "refund", True, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke software ke saath Shopify integrate karna hai", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapka dashboard mere browser mein sahi load nahi ho raha", "technical_support", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera account ka language change nahi ho raha", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki service ki pricing bahut confusing hai", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke yaha custom report ka feature chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera payment successful hai but subscription active nahi", "technical_support", True, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mera account ka plan downgrade kyun kiya", "account_issue", True, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapke software ke saath Zapier integration chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapki app ka customer support number do", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
    ("mera account ka profile picture update nahi ho raha", "account_issue", False, ["hinglish"], "v3 bulk hinglish"),
    ("aapne mujhe galat features ka price bata diya", "pricing", False, ["hinglish"], "v3 bulk hinglish"),
    ("mujhe aapki service ka referral link chahiye", "general_question", False, ["hinglish"], "v3 bulk hinglish"),
])

# More noisy variations
batch_add([
    ("i cnt figur out hw to use ur platform", "general_question", False, ["noisy"], "v3 bulk noisy"),
    ("ur chargin me evry mnth without my concent", "pricing", True, ["noisy"], "v3 bulk noisy"),
    ("the app is not wrking aftr the latest updat", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("i wnt a humen to hlp me with my accnt", "human_request", True, ["noisy"], "v3 bulk noisy"),
    ("hw much for the premium plan", "pricing", False, ["noisy"], "v3 bulk noisy"),
    ("my paymnt got deductd but i dnt see the features", "technical_support", True, ["noisy"], "v3 bulk noisy"),
    ("plz help im lockt out of my accnt since 2 days", "account_issue", True, ["noisy"], "v3 bulk noisy"),
    ("ur servis is the worst i ever had", "complaint", False, ["noisy"], "v3 bulk noisy"),
    ("i need my mony bak asap", "refund", True, ["noisy"], "v3 bulk noisy"),
    ("hw can i talk to a reel person", "human_request", True, ["noisy"], "v3 bulk noisy"),
    ("the search buton is not doin anything", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("my invoice is rong pls fix", "pricing", False, ["noisy"], "v3 bulk noisy"),
    ("wht is the price for aditional users", "pricing", False, ["noisy"], "v3 bulk noisy"),
    ("i forgat my pasword and the reset link is not wrking", "account_issue", False, ["noisy"], "v3 bulk noisy"),
    ("my team membrs cant see the files i shared", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("i wana buy ur premium plan but the cart is broken", "sales", True, ["noisy"], "v3 bulk noisy"),
    ("y is ur app so slo on my fone", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("u guys overchaarged me fix it", "pricing", True, ["noisy"], "v3 bulk noisy"),
    ("the butns on the dashboard r not clicable", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("i cnt delet my accnt from ur website help", "general_question", False, ["noisy"], "v3 bulk noisy"),
    ("pls help me my billing is rong", "pricing", False, ["noisy"], "v3 bulk noisy"),
    ("ur bot is not helpin at all gimme a person", "human_request", True, ["noisy"], "v3 bulk noisy"),
    ("the dasboard is not showin my corect data", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("i wnt my subscription canclled immedietly", "sales", False, ["noisy"], "v3 bulk noisy"),
    ("ur app keeps crashin evrytime i opn it", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("hw do i export my data frum ur platform", "general_question", False, ["noisy"], "v3 bulk noisy"),
    ("i m very angy about the billing issue refund now", "refund", True, ["noisy"], "v3 bulk noisy"),
    ("the integratin with our CRM is brkn since last week", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("wher can i find the setins for notifikashuns", "general_question", False, ["noisy"], "v3 bulk noisy"),
    ("my accnt was suspended for no reeson pls help", "account_issue", True, ["noisy"], "v3 bulk noisy"),
    ("i cnt chanj my plan on the websit", "sales", False, ["noisy"], "v3 bulk noisy"),
    ("ur team is usless i been waiting 5 days for a reply", "complaint", True, ["noisy"], "v3 bulk noisy"),
    ("hw 2 setup the integration with slack", "product_question", False, ["noisy"], "v3 bulk noisy"),
    ("my passwrd isnt wrking even after reset", "account_issue", False, ["noisy"], "v3 bulk noisy"),
    ("the app is drainin my battry too fast", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("i wnt a full refnd of my anual plan", "refund", True, ["noisy"], "v3 bulk noisy"),
    ("ur support team never replys on time", "complaint", False, ["noisy"], "v3 bulk noisy"),
    ("hw much 4 the team plan with 10 users", "pricing", False, ["noisy"], "v3 bulk noisy"),
    ("my account is showin someone elses details", "account_issue", True, ["noisy"], "v3 bulk noisy"),
    ("i need help but ur chatbot is useless", "human_request", True, ["noisy", "multi_intent"], "v3 bulk noisy"),
    ("the reports r not generatin since yesterdae", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("u guys charged me for a plan i dint sign up for", "pricing", True, ["noisy"], "v3 bulk noisy"),
    ("hw do i add team membrs to my workspace", "general_question", False, ["noisy"], "v3 bulk noisy"),
    ("my email notifcations arre not comin thru", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("i cnt login to my admin pannel", "account_issue", False, ["noisy"], "v3 bulk noisy"),
    ("ur premium plan is way overpriced for what u offer", "pricing", False, ["noisy"], "v3 bulk noisy"),
    ("the drag and drop isnt wrking in the builder", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("plz process my refund as promised", "refund", True, ["noisy"], "v3 bulk noisy"),
    ("i m havin truble with 2FA setup", "technical_support", False, ["noisy"], "v3 bulk noisy"),
    ("hw 2 upgrate my plan from basic to pro", "sales", False, ["noisy"], "v3 bulk noisy"),
])

# More escalation hard positives (subtle signals)
batch_add([
    ("I have been unable to work for 3 days because of this issue.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("My team is waiting on me and I can't deliver because of your bug.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("This is the second time this week the same error occurred.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I have a service level agreement and you are violating it.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("My bank statement shows a charge I did not authorize.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I require written confirmation of the steps you will take.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("This is causing a cascading failure in our infrastructure.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I am documenting every interaction for potential legal action.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("Our compliance team needs an immediate fix for this issue.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I need a root cause analysis for this recurring outage.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("Your error caused me to miss a critical business deadline.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I demand to know why this was not caught in your testing.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("My executive team is asking questions I cannot answer.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("This is a single point of failure for my entire operation.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I have already lost 12 hours of productivity this week.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("Multiple users in my organization are affected by this bug.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("This needs to be treated as a P0 incident immediately.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I am filing a formal complaint with your compliance department.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("My account was accessed without my knowledge or consent.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+"),
    ("You have exposed my personal data due to this security flaw.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I want a supervisor to personally review my case.", "human_request", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I have been a paying customer for 3 years and this is unacceptable.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("Your automated replies are not solving my specific situation.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I need an exception to your standard procedure for this case.", "human_request", True, ["escalation_positive"], "v3 bulk esc+"),
    ("This issue has financial implications that need to be addressed.", "complaint", True, ["escalation_positive"], "v3 bulk esc+"),
    ("My account details were changed by someone else. I need help.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I cannot proceed with my work until this is resolved.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("This is more urgent than my previous messages indicate.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
    ("I expect a personal phone call to discuss this matter.", "human_request", True, ["escalation_positive"], "v3 bulk esc+"),
    ("The workaround you suggested does not work for my use case.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+"),
])

# More escalation hard negatives
batch_add([
    ("No need to escalate. I was just asking for information.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("Please don't make a big deal out of this. Simple question.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I do not want this to be escalated to management.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("Forget I said anything about being upset. I'm fine now.", "complaint", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("Disregard my previous complaint. Everything is sorted.", "complaint", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("No action needed. I was just providing feedback.", "complaint", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("You don't need to call me about this. Email is enough.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("This isn't a support ticket. It's just a comment.", "other", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I already fixed the problem. No further assistance needed.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("Thanks for checking in but I resolved it myself.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("No follow up required. I was just giving a heads up.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I don't need anyone to intervene. Just wanted to let you know.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("Please do not assign this to your support team. Simple query.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I am not requesting any action. Just sharing my experience.", "complaint", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("No need to involve higher management. You can handle this.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I explicitly do not want a call. Text is fine.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("It's not that serious. Just answer when you get a chance.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("Don't worry about this. I'll figure it out myself.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I retract my earlier complaint. Everything is working.", "complaint", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("Please don't send anyone to my office. Just reply here.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("No need for a callback. I'm satisfied with the answer.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I was angry but I've cooled down. Don't escalate.", "complaint", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("This is resolved. Close the case.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("Don't take any action. I was just thinking out loud.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("My issue was answered in the FAQ. All set.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("Thanks for the help but I don't need anything else.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I don't consent to this being escalated. Keep it here.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("No need to transfer me. You answered my question perfectly.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-"),
    ("I was wrong about the issue. Everything actually works fine.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
    ("Please ignore. I solved it with the help article.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-"),
])

# More multi-intent
batch_add([
    ("I need to upgrade my plan but my payment method is failing.", "sales", True, ["multi_intent"], "v3 bulk multi"),
    ("Complaint about your service and I want to speak to a manager.", "human_request", True, ["multi_intent"], "v3 bulk multi"),
    ("My account was hacked and I need help securing it.", "account_issue", True, ["multi_intent"], "v3 bulk multi"),
    ("I want a refund and I also want to cancel my subscription.", "refund", True, ["multi_intent"], "v3 bulk multi"),
    ("Can you help me with a technical issue and also explain pricing?", "technical_support", False, ["multi_intent"], "v3 bulk multi"),
    ("Your product pricing is too high and your support is slow.", "pricing", False, ["multi_intent"], "v3 bulk multi"),
    ("I love the product but I'm having a login issue.", "account_issue", False, ["multi_intent"], "v3 bulk multi"),
    ("I need billing help and also want to change my plan.", "pricing", False, ["multi_intent"], "v3 bulk multi"),
    ("Your service is great but I'm locked out of my account.", "account_issue", False, ["multi_intent"], "v3 bulk multi"),
    ("I want to speak to sales and also need technical support.", "sales", False, ["multi_intent"], "v3 bulk multi"),
    ("Payment issue that needs a human to resolve.", "technical_support", True, ["multi_intent"], "v3 bulk multi"),
    ("I want to buy but first I need my account unlocked.", "sales", True, ["multi_intent"], "v3 bulk multi"),
    ("Complaint about pricing and request for refund.", "refund", True, ["multi_intent"], "v3 bulk multi"),
    ("I need technical help with integration and a price quote.", "technical_support", False, ["multi_intent"], "v3 bulk multi"),
    ("My account is acting weird and I'm thinking of canceling.", "account_issue", False, ["multi_intent"], "v3 bulk multi"),
    ("I want to upgrade but I'm frustrated with your support.", "sales", False, ["multi_intent"], "v3 bulk multi"),
    ("Can someone help me with my account? Also what plans do you have?", "account_issue", False, ["multi_intent"], "v3 bulk multi"),
    ("I'm interested in your enterprise plan but I need a demo first.", "sales", False, ["multi_intent"], "v3 bulk multi"),
    ("Technical issue with billing and I need a human to fix it.", "technical_support", True, ["multi_intent"], "v3 bulk multi"),
    ("Refund request for the annual plan I just bought.", "refund", True, ["multi_intent"], "v3 bulk multi"),
    ("I want to complain and also get a refund for poor service.", "refund", True, ["multi_intent"], "v3 bulk multi"),
    ("Help me with login and also tell me about premium features.", "account_issue", False, ["multi_intent"], "v3 bulk multi"),
    ("I'm angry about the price increase and I want options.", "pricing", False, ["multi_intent"], "v3 bulk multi"),
    ("My workflow is broken and I'm losing business because of it.", "technical_support", True, ["multi_intent"], "v3 bulk multi"),
    ("I have both a feature request and a bug to report.", "technical_support", False, ["multi_intent"], "v3 bulk multi"),
    ("Need help resetting my password and checking my plan.", "account_issue", False, ["multi_intent"], "v3 bulk multi"),
    ("Complaint about billing but resolved now thanks.", "complaint", False, ["multi_intent", "resolution_state"], "v3 bulk multi"),
    ("I want to buy the premium plan but your site crashed.", "sales", True, ["multi_intent"], "v3 bulk multi"),
    ("Mujhe refund chahiye aur account bhi close karna hai.", "refund", True, ["multi_intent", "hinglish"], "v3 bulk multi hinglish"),
    ("Mujhe aapki service se help chahiye aur price bhi puchhna hai.", "general_question", False, ["multi_intent", "hinglish"], "v3 bulk multi hinglish"),
    ("Mera account issue hai aur main bahut frustrated hu.", "account_issue", True, ["multi_intent", "hinglish"], "v3 bulk multi hinglish"),
    ("Mujhe technical support chahiye aur refund bhi.", "refund", True, ["multi_intent", "hinglish"], "v3 bulk multi hinglish"),
    ("Aapki pricing achi hai but features ke baare mein batao.", "product_question", False, ["multi_intent", "hinglish"], "v3 bulk multi hinglish"),
])

# Balance intents: more product_question, other, human_request, complaint, refund, general_question
batch_add([
    # product_question
    ("Do you offer a product tour?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I customize the notification templates?", "product_question", False, ["standard"], "v3 balance"),
    ("Does your software support Kanban boards?", "product_question", False, ["standard"], "v3 balance"),
    ("Is there a Gantt chart view?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I integrate with Microsoft Teams?", "product_question", False, ["standard"], "v3 balance"),
    ("Does the platform support recurring tasks?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I create project templates?", "product_question", False, ["standard"], "v3 balance"),
    ("Is there a time tracking module?", "product_question", False, ["standard"], "v3 balance"),
    ("Does it support dependency tracking?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I set resource allocation?", "product_question", False, ["standard"], "v3 balance"),
    ("Does the API support real-time events?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I export data in JSON format?", "product_question", False, ["standard"], "v3 balance"),
    ("Is there a built-in chat feature?", "product_question", False, ["standard"], "v3 balance"),
    ("Does your platform support SSO?", "product_question", False, ["standard"], "v3 balance"),
    ("Can I create custom approval workflows?", "product_question", False, ["standard"], "v3 balance"),
    # other
    ("This made my day!", "other", False, ["standard"], "v3 balance"),
    ("Having fun exploring your platform.", "other", False, ["standard"], "v3 balance"),
    ("Just browsing. Thanks!", "other", False, ["standard"], "v3 balance"),
    ("I don't need anything. Just saying hello.", "other", False, ["standard"], "v3 balance"),
    ("Your platform is interesting.", "other", False, ["standard"], "v3 balance"),
    ("Nice UI design!", "other", False, ["standard"], "v3 balance"),
    ("Good job on the recent update.", "other", False, ["standard"], "v3 balance"),
    ("I'm impressed with the onboarding flow.", "other", False, ["standard"], "v3 balance"),
    # human_request
    ("Please connect me to a real person immediately.", "human_request", True, ["standard"], "v3 balance"),
    ("I need actual human assistance not a knowledge base article.", "human_request", True, ["standard"], "v3 balance"),
    ("Let me speak to someone who understands my problem.", "human_request", True, ["standard"], "v3 balance"),
    ("This conversation needs a human touch. Transfer me.", "human_request", True, ["standard"], "v3 balance"),
    ("I'm not going to explain this to another bot. Human now.", "human_request", True, ["standard"], "v3 balance"),
    ("Can I call someone directly at your office?", "human_request", True, ["standard"], "v3 balance"),
    ("I need a person who can make decisions.", "human_request", True, ["standard"], "v3 balance"),
    ("Stop the automated messages. Give me a support agent.", "human_request", True, ["standard"], "v3 balance"),
    ("Connect me to your emergency support line.", "human_request", True, ["standard"], "v3 balance"),
    ("I want to talk to a specialist, not a general agent.", "human_request", True, ["standard"], "v3 balance"),
    # refund
    ("I want to return the product and get a full refund.", "refund", True, ["standard"], "v3 balance"),
    ("Please cancel my subscription and refund the remaining amount.", "refund", True, ["standard"], "v3 balance"),
    ("I was charged for a plan I never signed up for. Refund now.", "refund", True, ["standard"], "v3 balance"),
    ("Your service didn't work as advertised. I want my money back.", "refund", True, ["standard"], "v3 balance"),
    ("Process my refund request number RF-2024-001.", "refund", True, ["standard"], "v3 balance"),
    ("I demand a full refund including the setup and training fees.", "refund", True, ["standard"], "v3 balance"),
    ("Refund my money or I will share my experience on social media.", "refund", True, ["standard"], "v3 balance"),
    ("You billed me incorrectly for 3 months. Refund the difference.", "refund", True, ["standard"], "v3 balance"),
    # complaint
    ("I'm disappointed with the lack of features in the basic plan.", "complaint", False, ["standard"], "v3 balance"),
    ("Your customer support is unresponsive and unhelpful.", "complaint", False, ["standard"], "v3 balance"),
    ("I feel like your company doesn't value existing customers.", "complaint", False, ["standard"], "v3 balance"),
    ("The quality of your service has declined significantly.", "complaint", False, ["standard"], "v3 balance"),
    ("Your platform has too many bugs for a production service.", "complaint", False, ["standard"], "v3 balance"),
    ("I regret choosing your platform over competitors.", "complaint", False, ["standard"], "v3 balance"),
    ("Your sales team promised features that don't exist.", "complaint", False, ["standard"], "v3 balance"),
    # general_question
    ("How often do you release new features?", "general_question", False, ["standard"], "v3 balance"),
    ("Is there a way to request new features?", "general_question", False, ["standard"], "v3 balance"),
    ("Can I participate in beta testing?", "general_question", False, ["standard"], "v3 balance"),
    ("How do you handle data privacy?", "general_question", False, ["standard"], "v3 balance"),
    ("What is your company's mission?", "general_question", False, ["standard"], "v3 balance"),
    ("Do you have offices in Asia?", "general_question", False, ["standard"], "v3 balance"),
    ("Can I use your service for personal projects?", "general_question", False, ["standard"], "v3 balance"),
    ("What makes your service different from competitors?", "general_question", False, ["standard"], "v3 balance"),
    ("Do you have a loyalty program for long-term customers?", "general_question", False, ["standard"], "v3 balance"),
    ("Is your platform accessible for users with disabilities?", "general_question", False, ["standard"], "v3 balance"),
])

# ============================================================
# BULK ADDITION VIA TEMPLATES (round 2 - more volume)
# ============================================================

# General questions - bulk round 2
gq_templates2 = [
    "How do I set up {feature}?",
    "Can I customize {feature}?",
    "Is there a way to {feature}?",
    "How does {feature} affect my account?",
    "What is the purpose of {feature}?",
]
gq_features2 = [
    "billing alerts", "auto-renewal", "team permissions",
    "data retention limits", "profile visibility", "email digests",
    "activity logs", "session timeouts", "password policies",
    "account recovery", "notification preferences", "dashboard widgets",
    "custom branding", "user invitations", "bulk operations",
    "search filters", "keyboard shortcuts", "accessibility options",
    "import features", "export options",
]
for t in gq_templates2:
    for f in gq_features2:
        add(t.format(feature=f), "general_question", False, ["standard"], "v3 bulk gq2")
        if len(examples) - v1_count > 3400:
            break
    if len(examples) - v1_count > 3400:
        break

# Pricing templates - more volume
p_templates2 = [
    "Tell me about pricing for {feature}.",
    "What are the costs associated with {feature}?",
    "How is {feature} priced?",
    "Do you offer different pricing tiers for {feature}?",
]
p_features2 = [
    "data storage", "API calls", "user accounts",
    "team collaboration", "premium features", "compliance features",
    "audit trail", "custom integrations", "dedicated support",
    "training sessions", "onboarding assistance", "data migration",
]
for t in p_templates2:
    for f in p_features2:
        add(t.format(feature=f), "pricing", False, ["standard", "pricing"], "v3 bulk pricing2")
        if len(examples) - v1_count > 3600:
            break
    if len(examples) - v1_count > 3600:
        break

# Technical support - more templates
ts_templates2 = [
    "{feature} is malfunctioning.",
    "Having issues with {feature}.",
    "{feature} stopped responding.",
    "{feature} is causing errors in my workflow.",
    "The {feature} feature seems to be broken.",
]
ts_features2 = [
    "bulk edit", "user import", "file preview",
    "email parser", "ticket routing", "auto-responder",
    "knowledge base search", "customer portal",
    "report scheduling", "dashboard export",
    "data filtering", "sort functionality",
    "batch processing", "real-time updates",
]
for t in ts_templates2:
    for f in ts_features2:
        add(t.format(feature=f), "technical_support", False, ["standard", "technical"], "v3 bulk tech2")
        if len(examples) - v1_count > 3800:
            break
    if len(examples) - v1_count > 3800:
        break

# Additional confusion pairs
add("Does your premium plan have unlimited storage?", "product_question", False, ["standard"], "v3 bulk confusion")
add("What is included in your premium offering?", "general_question", False, ["standard"], "v3 bulk confusion")
add("Can your API handle webhook retries automatically?", "product_question", False, ["standard"], "v3 bulk confusion")
add("How does your error handling work?", "general_question", False, ["standard"], "v3 bulk confusion")
add("Is two-factor authentication available on all plans?", "product_question", False, ["standard"], "v3 bulk confusion")
add("What authentication methods do you support?", "general_question", False, ["standard"], "v3 bulk confusion")
add("Can I use your platform to send transactional emails?", "product_question", False, ["standard"], "v3 bulk confusion")
add("Do you have email sending capabilities?", "general_question", False, ["standard"], "v3 bulk confusion")
add("Does your dashboard support real-time data visualization?", "product_question", False, ["standard"], "v3 bulk confusion")
add("What kind of charts and graphs do you offer?", "general_question", False, ["standard"], "v3 bulk confusion")
add("My data export keeps failing with no error message", "technical_support", False, ["standard"], "v3 bulk confusion")
add("Your export function is completely unreliable", "complaint", False, ["standard"], "v3 bulk confusion")
add("The API returns HTTP 500 for valid requests", "technical_support", True, ["standard"], "v3 bulk confusion")
add("Your API is poorly designed and always breaks", "complaint", False, ["standard"], "v3 bulk confusion")
add("I keep getting session timeouts every 5 minutes", "technical_support", False, ["standard"], "v3 bulk confusion")
add("Your platform logs me out constantly and it's maddening", "complaint", False, ["standard"], "v3 bulk confusion")
add("The price of the enterprise plan is too high", "pricing", False, ["standard"], "v3 bulk confusion")
add("I want to purchase the enterprise plan for my company", "sales", False, ["standard"], "v3 bulk confusion")
add("What is the annual cost for 100 users?", "pricing", False, ["standard"], "v3 bulk confusion")
add("Please set up 100 users on the annual enterprise plan", "sales", False, ["standard"], "v3 bulk confusion")
add("Talk to a human about my account problem", "human_request", True, ["standard"], "v3 bulk confusion")
add("I want to discuss my account issues with sales", "sales", False, ["standard"], "v3 bulk confusion")
add("I need a person to help me with this", "human_request", True, ["standard"], "v3 bulk confusion")
add("I need a person to help me purchase", "sales", False, ["standard"], "v3 bulk confusion")

# Additional escalation hard negatives with varied language
add("I get that you want to help but do not escalate this.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("No escalation necessary, I just had a quick question.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("Don't transfer me anywhere. Just answer the question.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("I'm not looking for a solution. Just sharing my thoughts.", "complaint", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("Please, no follow up calls. I hate phone calls.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("Do not contact me. I will contact you if I need help.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("Resolved. Thanks, close this out.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-")
add("I am all good now. No need to do anything.", "general_question", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-")
add("False alarm. Everything works. Sorry for the trouble.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-")
add("Don't worry, I figured out the issue. All good.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-")
add("Nothing to see here. Just a test message.", "other", False, ["hard_negative_escalation"], "v3 bulk esc-")
add("I was frustrated but it's all working. No escalation.", "complaint", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-")
add("Turns out I was wrong. Your platform works perfectly.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-")
add("Mai theek hu. Koi action mat lo.", "general_question", False, ["hard_negative_escalation", "hinglish"], "v3 bulk esc-")
add("Sab sahi hai. Close karo ticket.", "technical_support", False, ["hard_negative_escalation", "resolution_state", "hinglish"], "v3 bulk esc-")
add("Koi zaroorat nahi hai escalate karne ki. Simple baat hai.", "general_question", False, ["hard_negative_escalation", "hinglish"], "v3 bulk esc-")

# Additional escalation hard positives
add("This is a production down scenario. Page the on-call team.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("I need an engineer to look at this immediately.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("My entire company's workflow is halted by this bug.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("This is causing a data integrity issue that could be catastrophic.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("I need a hotfix deployed to production right now.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("This has financial reporting implications for our quarter close.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("I am formally requesting an escalation to your CTO office.", "complaint", True, ["escalation_positive"], "v3 bulk esc+")
add("This security vulnerability needs to be patched immediately.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("I have identified a critical flaw in your payment processing.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("My account seems to have been accessed by an unauthorized user.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+")
add("This is a data privacy concern that requires urgent attention.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+")
add("I am unable to proceed with my business operations due to this.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("I need someone to personally guarantee this will not happen again.", "complaint", True, ["escalation_positive"], "v3 bulk esc+")
add("Your platform failure has caused reputational damage to my company.", "complaint", True, ["escalation_positive"], "v3 bulk esc+")
add("I am logging this as an official incident per our contract terms.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+")
add("I expect compensation for the downtime I have experienced.", "refund", True, ["escalation_positive"], "v3 bulk esc+")
add("Maine bahut patience dikhaya but ab limit aa gayi. Senior ko bulao.", "complaint", True, ["escalation_positive", "hinglish"], "v3 bulk esc+")
add("Mera poora kaam ruka hai. Yeh urgent hai. Koi action lo.", "technical_support", True, ["escalation_positive", "hinglish"], "v3 bulk esc+")
add("Aapki vajah se mera client loss ho raha hai. Ab legal action lunga.", "complaint", True, ["escalation_positive", "hinglish"], "v3 bulk esc+")

# Additional Hinglish
add("aapki app mein error aa raha hai fix karo jaldi", "technical_support", True, ["hinglish"], "v3 bulk hinglish")
add("mujhe aapke yaha par existing customer discount chahiye", "sales", False, ["hinglish"], "v3 bulk hinglish")
add("mera account ka 2FA setup nahi ho raha", "account_issue", False, ["hinglish"], "v3 bulk hinglish")
add("aapki pricing page par GST clear nahi hai", "pricing", False, ["hinglish"], "v3 bulk hinglish")
add("mujhe aapke software ke saath WhatsApp integration chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish")
add("Mera support ticket 3 din se unresolved hai", "complaint", True, ["hinglish"], "v3 bulk hinglish")
add("aapke yaha bulk SMS kitne ka hai", "pricing", False, ["hinglish"], "v3 bulk hinglish")
add("mujhe aapki service se bahut help mili thanks", "other", False, ["hinglish"], "v3 bulk hinglish")
add("aapke yaha koi loyalty reward program hai kya", "general_question", False, ["hinglish"], "v3 bulk hinglish")
add("mera account mein kuch settings theek nahi hain", "account_issue", False, ["hinglish"], "v3 bulk hinglish")
add("aapki app ka latest version kab aayega", "product_question", False, ["hinglish"], "v3 bulk hinglish")
add("mujhe aapke yaha bulk data import karna hai", "general_question", False, ["hinglish"], "v3 bulk hinglish")
add("mera plan upgrade ka payment fail ho raha hai", "sales", True, ["hinglish"], "v3 bulk hinglish")
add("aapne mujhe refund ka promise kiya tha but nahi diya", "refund", True, ["hinglish"], "v3 bulk hinglish")
add("aapki app ka response time bahut slow hai", "technical_support", False, ["hinglish"], "v3 bulk hinglish")
add("mujhe aapke software mein custom field add karne hain", "product_question", False, ["hinglish"], "v3 bulk hinglish")
add("mera account ka primary email change karo", "account_issue", False, ["hinglish"], "v3 bulk hinglish")
add("aapka pricing structure bahut complicated hai", "pricing", False, ["hinglish"], "v3 bulk hinglish")
add("mujhe aapki service ka live demo dikhao", "sales", False, ["hinglish"], "v3 bulk hinglish")
add("aapke integration se hame bahut help mili", "other", False, ["hinglish"], "v3 bulk hinglish")

# Additional noisy
add("plz help my bill is way 2 high", "pricing", False, ["noisy"], "v3 bulk noisy")
add("i cnt figur out this app at all", "general_question", False, ["noisy"], "v3 bulk noisy")
add("ur team charged me twice this mnth fix it", "pricing", True, ["noisy"], "v3 bulk noisy")
add("the export btn is greyed out i cant click it", "technical_support", False, ["noisy"], "v3 bulk noisy")
add("hw come ur premium plan is so expensiv", "pricing", False, ["noisy"], "v3 bulk noisy")
add("my teammates cnt acess the folder i shared", "technical_support", False, ["noisy"], "v3 bulk noisy")
add("i want my mony back its been 10 dys", "refund", True, ["noisy"], "v3 bulk noisy")
add("wht do i do if i 4get my username", "general_question", False, ["noisy"], "v3 bulk noisy")
add("the updat made things worse not bettr", "complaint", False, ["noisy"], "v3 bulk noisy")
add("can u help me setup my account plz", "account_issue", False, ["noisy"], "v3 bulk noisy")
add("i cnt find the dark mode option anywer", "product_question", False, ["noisy"], "v3 bulk noisy")
add("ur app is too complic8ed simplify it", "complaint", False, ["noisy"], "v3 bulk noisy")
add("the graph on my dashbord is not loadin", "technical_support", False, ["noisy"], "v3 bulk noisy")
add("i need a cal back asap its urgent", "human_request", True, ["noisy"], "v3 bulk noisy")
add("y did u suspend my accnt without reason", "account_issue", True, ["noisy"], "v3 bulk noisy")

# Additional negation
add("I'm not looking for a refund. I'm looking for an apology.", "complaint", False, ["negation"], "v3 bulk negation")
add("Don't give me a refund. I want the service to work.", "technical_support", True, ["negation"], "v3 bulk negation")
add("I do not want to cancel. I just want to understand my bill.", "pricing", False, ["negation"], "v3 bulk negation")
add("No need to apologize. Just fix the problem.", "technical_support", False, ["negation"], "v3 bulk negation")
add("I'm not leaving your platform. I'm just frustrated.", "complaint", False, ["negation"], "v3 bulk negation")
add("I don't want compensation. I want a solution.", "technical_support", True, ["negation"], "v3 bulk negation")
add("This is not a complaint. This is constructive feedback.", "other", False, ["negation"], "v3 bulk negation")
add("Not urgent. Please don't escalate this.", "general_question", False, ["negation"], "v3 bulk negation")
add("Mujhe koi call nahi karna. Sirf jawab do.", "general_question", False, ["negation", "hinglish"], "v3 bulk negation")
add("Refund nahi chahiye mujhe. Service do sahi se.", "technical_support", True, ["negation", "hinglish"], "v3 bulk negation")

# Additional resolution state
add("I fixed the issue by restarting my browser. Thanks anyway.", "technical_support", False, ["resolution_state"], "v3 bulk resolution")
add("The problem went away after I cleared my cache.", "technical_support", False, ["resolution_state"], "v3 bulk resolution")
add("My colleague helped me resolve this. Close the ticket.", "technical_support", False, ["resolution_state"], "v3 bulk resolution")
add("The patch you sent fixed the issue. Thank you!", "technical_support", False, ["resolution_state"], "v3 bulk resolution")
add("I was upset but your team fixed everything. All good now.", "complaint", False, ["resolution_state"], "v3 bulk resolution")
add("The refund appeared in my account. Thanks for processing.", "refund", False, ["resolution_state"], "v3 bulk resolution")
add("Everything is back to normal after the update.", "technical_support", False, ["resolution_state"], "v3 bulk resolution")
add("My access was restored. Thanks for the quick help.", "account_issue", False, ["resolution_state"], "v3 bulk resolution")
add("The billing issue has been resolved on your end. Thanks.", "pricing", False, ["resolution_state"], "v3 bulk resolution")
add("I managed to get the feature working. No more help needed.", "technical_support", False, ["resolution_state"], "v3 bulk resolution")

# Additional information vs action
add("Can you tell me how to upgrade?", "general_question", False, ["standard"], "v3 bulk infoact")
add("Upgrade my account to premium right now.", "sales", False, ["standard"], "v3 bulk infoact")
add("What is your cancellation policy?", "general_question", False, ["standard"], "v3 bulk infoact")
add("Cancel my subscription effective immediately.", "sales", False, ["standard"], "v3 bulk infoact")
add("How does the refund process work?", "general_question", False, ["standard"], "v3 bulk infoact")
add("Process my refund request number 12345.", "refund", True, ["standard"], "v3 bulk infoact")
add("Explain the steps to reset my password.", "general_question", False, ["standard"], "v3 bulk infoact")
add("Reset my password right now I'm locked out.", "account_issue", True, ["standard"], "v3 bulk infoact")
add("What are my options for premium support?", "general_question", False, ["standard"], "v3 bulk infoact")
add("Activate premium support for my account now.", "sales", False, ["standard"], "v3 bulk infoact")
add("Tell me about your data backup policy.", "general_question", False, ["standard"], "v3 bulk infoact")
add("Restore my data from yesterday's backup.", "technical_support", True, ["standard"], "v3 bulk infoact")

# ============================================================
# BULK ADDITION ROUND 3 - TEMPLATES
# ============================================================

gq_templates3 = [
    "Can you guide me through {feature}?",
    "How do I configure {feature}?",
    "I need information about {feature}.",
    "What should I know about {feature}?",
]
gq_features3 = [
    "the account setup", "payment methods", "billing history",
    "subscription management", "team collaboration", "file organization",
    "data sharing", "privacy settings", "security options",
    "the notification center", "activity tracking", "usage reports",
    "the contact directory", "the file manager", "user management",
]
for t in gq_templates3:
    for f in gq_features3:
        add(t.format(feature=f), "general_question", False, ["standard"], "v3 bulk gq3")
        if len(examples) - v1_count > 2900:
            break
    if len(examples) - v1_count > 2900:
        break

ts_templates3 = [
    "Encountering problems with {feature}.",
    "{feature} is giving me trouble.",
    "I am unable to use {feature} correctly.",
    "{feature} has a bug that needs fixing.",
]
ts_features3 = [
    "the tag system", "the folder structure", "the template editor",
    "the color picker", "the font selector", "the alignment tool",
    "the grid view", "the list view", "the calendar view",
    "the timeline view", "the attachment preview", "the inline editor",
    "the sidebar navigation", "the search filter", "the sort option",
]
for t in ts_templates3:
    for f in ts_features3:
        add(t.format(feature=f), "technical_support", False, ["standard", "technical"], "v3 bulk tech3")
        if len(examples) - v1_count > 3100:
            break
    if len(examples) - v1_count > 3100:
        break

# More hand-crafted Hinglish
batch_add([
    ("aapka software mere browser mein sahi load nahi ho raha", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapki app mein ek feature suggest karna hai", "other", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account ki security settings change karni hain", "account_issue", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki app par file upload nahi ho raha error de raha hai", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke yaha bulk user import ka feature chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapne mujhe galat currency mein payment kiya", "pricing", True, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapki service ka 1 mahine ka free trial chahiye", "sales", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account mein profile update nahi ho raha", "account_issue", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki app bahut storage le rahi hai phone ki", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("kya main aapke platform par hindi mein type kar sakta hu", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke yaha auto-generated invoice chahiye", "general_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapne mera support ticket close kyun kar diya", "complaint", True, ["hinglish"], "v3 bulk hinglish3"),
    ("mera payment successful hai but account mein credit nahi aaya", "technical_support", True, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapki service ke through bulk email bhejne hain", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapke yaha custom domain kitne ka milta hai", "pricing", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account ka notification sound kaam nahi kar raha", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapki app mein team member ka role change karna hai", "account_issue", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki service se bahut satisfaction hai thank you", "other", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera plan renewal ka reminder bhejo", "general_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapke yaha par GST invoice kaise milega", "pricing", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke software ke saath Google Sheets sync chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account ka storage full ho gaya hai upgrade karo", "sales", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki app ka backup feature kaise kaam karta hai", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke yaha dedicated account manager chahiye", "sales", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapne mujhe wrong invoice bheja hai dobara bhejo", "pricing", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera API integration kaam nahi kar raha help karo", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki app ki performance improve karo bahut slow hai", "complaint", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke yaha bulk discount chahiye 200 users ke liye", "sales", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account ka two-factor authentication reset karo", "account_issue", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki app mein Hindi language option kab aayega", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke software ka trial extend karna hai", "sales", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapka pricing page mujhe samajh nahi aa raha", "pricing", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account ki billing cycle change karo", "pricing", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki team ka response time bahut acha hai", "other", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke software ke saath Shopify integration chahiye", "product_question", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera invoice download karte waqt error aa raha hai", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("aapki app par mujhe koi error aa raha hai fix karo", "technical_support", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mujhe aapke yaha se certificate chahiye service ke liye", "other", False, ["hinglish"], "v3 bulk hinglish3"),
    ("mera account delete karne ke baad bhi charge aa raha hai", "refund", True, ["hinglish"], "v3 bulk hinglish3"),
    ("aapne mujhe galat plan recommend kiya tha", "complaint", False, ["hinglish"], "v3 bulk hinglish3"),
])

# More noisy
batch_add([
    ("pls help my accnt has been suspndd unfairly", "account_issue", True, ["noisy"], "v3 bulk noisy3"),
    ("wht is the process for cancelling my subscription", "general_question", False, ["noisy"], "v3 bulk noisy3"),
    ("u guyz charged me for a plan i dint order", "pricing", True, ["noisy"], "v3 bulk noisy3"),
    ("the report generator is nt wrking properly", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("hw do i get a refund for the last month", "refund", False, ["noisy"], "v3 bulk noisy3"),
    ("i cnt find the upgarde option anywher in my accnt", "sales", False, ["noisy"], "v3 bulk noisy3"),
    ("ur platform is so confusin i dnt get it", "general_question", False, ["noisy"], "v3 bulk noisy3"),
    ("my accnt details were changed without my knwledge", "account_issue", True, ["noisy"], "v3 bulk noisy3"),
    ("the color of my dashboard is all messed up after update", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("i want 2 talk 2 a human now stop the bot", "human_request", True, ["noisy"], "v3 bulk noisy3"),
    ("the payment page is nt loading on mobil", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("hw long does refnd take to process", "refund", False, ["noisy"], "v3 bulk noisy3"),
    ("i cnt see my invoices in the billng section", "pricing", False, ["noisy"], "v3 bulk noisy3"),
    ("ur servis quality went downhil since last year", "complaint", False, ["noisy"], "v3 bulk noisy3"),
    ("the setup wizard is stuck at step 3 pls help", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("i paid for premium but got standrd features", "account_issue", True, ["noisy"], "v3 bulk noisy3"),
    ("the search feature is not returnin any results", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("i need 2 spek to the billing dept urgently", "human_request", True, ["noisy"], "v3 bulk noisy3"),
    ("my team cant acess the project i created", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("y is your customer servis so bad", "complaint", False, ["noisy"], "v3 bulk noisy3"),
    ("i dnt knw why my payment faild pls check", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("the drag feature is not wrking on the dashbrd", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("i wnt to buy your product but need info first", "product_question", False, ["noisy"], "v3 bulk noisy3"),
    ("ur app is crashing repeatedly fix it plz", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("i cnt remember my login id help", "account_issue", False, ["noisy"], "v3 bulk noisy3"),
    ("the installer is failng every time i try", "technical_support", False, ["noisy"], "v3 bulk noisy3"),
    ("hw do i connect ur app to my email", "general_question", False, ["noisy"], "v3 bulk noisy3"),
    ("my profil pictur is not updating", "account_issue", False, ["noisy"], "v3 bulk noisy3"),
    ("u overcharged me for the last 3 months fix it", "pricing", True, ["noisy"], "v3 bulk noisy3"),
    ("the app interface is not intuitive at all", "complaint", False, ["noisy"], "v3 bulk noisy3"),
])

# More escalation positives (targeting subtle signals)
batch_add([
    ("I cannot access critical files for a presentation in 1 hour.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("Your system failure caused me to miss an important deadline.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("This is the third outage this month. I demand reliability.", "complaint", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("I have tried every possible solution and nothing works.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("My business is losing $500 per hour this system is down.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("Your support has been unacceptably slow for a paid plan.", "complaint", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("I need a dedicated resource assigned to fix this problem.", "human_request", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("This issue is causing a compliance risk for my company.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("I'm being asked by my management for an update on this issue.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("Your error has cost me real money. I expect compensation.", "refund", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("I need a call from senior management about this issue.", "human_request", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("This has been going on for two weeks with no end in sight.", "complaint", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("I am unable to serve my own customers because of your bug.", "technical_support", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("My user account was deleted erroneously and I need it back.", "account_issue", True, ["escalation_positive"], "v3 bulk esc+3"),
    ("I am considering legal recourse if this is not fixed now.", "complaint", True, ["escalation_positive"], "v3 bulk esc+3"),
])

# More escalation hard negatives
batch_add([
    ("All good on my end now. No further action necessary.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-3"),
    ("No need for any intervention. Just wanted to report a minor thing.", "technical_support", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("I solved the problem using your documentation. Great docs!", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-3"),
    ("Do not escalate. This is not a support issue. Just feedback.", "complaint", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("Please do not contact me. I will reach out if needed.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("My previous message was premature. Everything is fine.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-3"),
    ("No help needed. I was just exploring the interface.", "other", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("Don't take any action on my account. Just had a question.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("No call required. A simple text answer will do.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("I am fine. No follow up needed. Thanks.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("Cancel my earlier request. Everything resolved itself.", "technical_support", False, ["hard_negative_escalation", "resolution_state"], "v3 bulk esc-3"),
    ("I don't want a call back. Just confirm via email.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("No escalation. Simple thing. Don't overcomplicate it.", "general_question", False, ["hard_negative_escalation"], "v3 bulk esc-3"),
    ("Maine apni problem khud solve kar li. Koi action na lo.", "technical_support", False, ["hard_negative_escalation", "resolution_state", "hinglish"], "v3 bulk esc-3"),
    ("Bas puchh raha tha. Koi escalate nahi karna.", "general_question", False, ["hard_negative_escalation", "hinglish"], "v3 bulk esc-3"),
])

# More multi-intent with diverse patterns
batch_add([
    ("I need technical help with the API and also a pricing question.", "technical_support", False, ["multi_intent"], "v3 bulk multi3"),
    ("Your service is bad and I want to cancel my subscription.", "complaint", False, ["multi_intent"], "v3 bulk multi3"),
    ("I want to complain about billing and get it fixed.", "pricing", False, ["multi_intent"], "v3 bulk multi3"),
    ("My account is compromised and I need immediate help.", "account_issue", True, ["multi_intent"], "v3 bulk multi3"),
    ("Can I get a refund and also report a security issue?", "refund", True, ["multi_intent"], "v3 bulk multi3"),
    ("I have a question about features and a problem with login.", "account_issue", False, ["multi_intent"], "v3 bulk multi3"),
    ("I'm both happy with the product and frustrated with support.", "complaint", False, ["multi_intent"], "v3 bulk multi3"),
    ("Need a human to help with purchase and also technical setup.", "sales", False, ["multi_intent"], "v3 bulk multi3"),
    ("Your app keeps crashing and I want to speak to someone.", "technical_support", True, ["multi_intent"], "v3 bulk multi3"),
    ("Payment issue and account lockout at the same time.", "account_issue", True, ["multi_intent"], "v3 bulk multi3"),
    ("I want to upgrade but I'm also locked out of my account.", "sales", True, ["multi_intent"], "v3 bulk multi3"),
    ("Complaint about pricing and need technical help.", "pricing", False, ["multi_intent"], "v3 bulk multi3"),
    ("Refund request for poor service and also want to cancel.", "refund", True, ["multi_intent"], "v3 bulk multi3"),
    ("Your platform is great but I have a few feature requests.", "product_question", False, ["multi_intent"], "v3 bulk multi3"),
    ("Mujhe refund chahiye aur technical support bhi.", "refund", True, ["multi_intent", "hinglish"], "v3 bulk multi3"),
    ("Mera account issue hai aur main bahut pareshan hu.", "account_issue", True, ["multi_intent", "hinglish"], "v3 bulk multi3"),
    ("Aapki pricing achi hai but features ke baare mein batao.", "product_question", False, ["multi_intent", "hinglish"], "v3 bulk multi3"),
    ("Mujhe aapke yaha se plan lena hai but pehle demo chahiye.", "sales", False, ["multi_intent", "hinglish"], "v3 bulk multi3"),
])

# Balance with more of underrepresented intents
batch_add([
    # product_question
    ("Does the API support batch operations?", "product_question", False, ["standard"], "v3 balance3"),
    ("Can I set up conditional logic in forms?", "product_question", False, ["standard"], "v3 balance3"),
    ("Is there a drag-and-drop form builder?", "product_question", False, ["standard"], "v3 balance3"),
    ("Does your platform support multi-factor authentication?", "product_question", False, ["standard"], "v3 balance3"),
    ("Can I create custom email templates?", "product_question", False, ["standard"], "v3 balance3"),
    ("Is there a built-in reporting tool?", "product_question", False, ["standard"], "v3 balance3"),
    ("Can I integrate with Google Analytics?", "product_question", False, ["standard"], "v3 balance3"),
    # other
    ("Just saying hi, love your work!", "other", False, ["standard"], "v3 balance3"),
    ("Cool feature set you have there.", "other", False, ["standard"], "v3 balance3"),
    ("Nice, thanks!", "other", False, ["standard"], "v3 balance3"),
    ("I appreciate the quick response.", "other", False, ["standard"], "v3 balance3"),
    # human_request
    ("I need a real person to help with this billing dispute.", "human_request", True, ["standard"], "v3 balance3"),
    ("Stop the automated messages. I want to talk to a human.", "human_request", True, ["standard"], "v3 balance3"),
    ("Your virtual assistant is not helping. Connect me to a person.", "human_request", True, ["standard"], "v3 balance3"),
    ("Can you please connect me to a real support agent?", "human_request", True, ["standard"], "v3 balance3"),
    # complaint
    ("I'm extremely dissatisfied with your platform's reliability.", "complaint", False, ["standard"], "v3 balance3"),
    ("Your product quality has degraded noticeably over time.", "complaint", False, ["standard"], "v3 balance3"),
    ("This is false advertising. Your product does not match claims.", "complaint", False, ["standard"], "v3 balance3"),
    # refund
    ("I want a full refund because your service never worked.", "refund", True, ["standard"], "v3 balance3"),
    ("Refund my money immediately or I will file a complaint.", "refund", True, ["standard"], "v3 balance3"),
    ("I demand a refund for the features that were promised but missing.", "refund", True, ["standard"], "v3 balance3"),
    # general_question
    ("Can I change my plan billing frequency?", "general_question", False, ["standard"], "v3 balance3"),
    ("How do I view my team's activity log?", "general_question", False, ["standard"], "v3 balance3"),
    ("Is there a limit to how many people I can invite?", "general_question", False, ["standard"], "v3 balance3"),
    ("How do I check my current plan details?", "general_question", False, ["standard"], "v3 balance3"),
])

# ============================================================
# BULK ADDITION ROUND 4 - FINAL PUSH TO 2000+ NEW
# ============================================================

# More confusion pairs
batch_add([
    ("What's the difference between your plans?", "general_question", False, ["confusion_pair"], "v4 confusion"),
    ("Does the premium plan have everything the pro plan has?", "product_question", False, ["confusion_pair"], "v4 confusion"),
    ("I need someone to help me fix my account", "account_issue", False, ["confusion_pair"], "v4 confusion"),
    ("I need someone to help me buy the right plan", "sales", False, ["confusion_pair"], "v4 confusion"),
    ("Your app crashes every time I try to export data", "technical_support", False, ["confusion_pair"], "v4 confusion"),
    ("Your app is completely unusable after the update", "complaint", False, ["confusion_pair"], "v4 confusion"),
    ("Tell me the monthly cost for 10 users", "pricing", False, ["confusion_pair"], "v4 confusion"),
    ("I want to get my team started with your product", "sales", False, ["confusion_pair"], "v4 confusion"),
    ("Can a real person help me with my account?", "human_request", True, ["confusion_pair"], "v4 confusion"),
    ("I want to discuss my requirements with a salesperson", "sales", False, ["confusion_pair"], "v4 confusion"),
    ("What happens if I don't use your service for a month?", "general_question", False, ["confusion_pair"], "v4 confusion"),
    ("Does your platform automatically pause inactive accounts?", "product_question", False, ["confusion_pair"], "v4 confusion"),
])

# Short form / single word additions
batch_add([
    ("Price?", "pricing", False, ["standard"], "v4 short"),
    ("Cost for enterprise?", "pricing", False, ["standard"], "v4 short"),
    ("Need human urgently.", "human_request", True, ["standard"], "v4 short"),
    ("Refund request.", "refund", True, ["standard"], "v4 short"),
    ("Account locked.", "account_issue", True, ["standard"], "v4 short"),
    ("Login failed.", "account_issue", False, ["standard"], "v4 short"),
    ("Error 403.", "technical_support", False, ["standard"], "v4 short"),
    ("Payment declined.", "technical_support", False, ["standard"], "v4 short"),
    ("Upgrade please.", "sales", False, ["standard"], "v4 short"),
    ("Cancel now.", "sales", False, ["standard"], "v4 short"),
    ("Too slow.", "technical_support", False, ["standard"], "v4 short"),
    ("Very bad experience.", "complaint", False, ["standard"], "v4 short"),
    ("Excellent support!", "other", False, ["standard"], "v4 short"),
    ("Hinglish: price batao.", "pricing", False, ["hinglish"], "v4 short"),
    ("Hinglish: refund do.", "refund", True, ["hinglish"], "v4 short"),
    ("Hinglish: help karo.", "general_question", False, ["hinglish"], "v4 short"),
    ("Hinglish: human do.", "human_request", True, ["hinglish"], "v4 short"),
    ("Hinglish: band karo.", "sales", False, ["hinglish"], "v4 short"),
    ("Hinglish: error hai.", "technical_support", False, ["hinglish"], "v4 short"),
    ("Hinglish: nahi chal raha.", "technical_support", False, ["hinglish"], "v4 short"),
])

# More Hinglish natural phrases
batch_add([
    ("aapke yaha kya kya features available hain", "product_question", False, ["hinglish"], "v4 hinglish"),
    ("mujhe aapki service ka refund chahiye", "refund", True, ["hinglish"], "v4 hinglish"),
    ("mera account ka kuch gadbad hai check karo", "account_issue", False, ["hinglish"], "v4 hinglish"),
    ("aapki app mein koi bug hai fix karo jaldi", "technical_support", False, ["hinglish"], "v4 hinglish"),
    ("mujhe aapke yaha par job karni hai", "other", False, ["hinglish"], "v4 hinglish"),
    ("aapka software kaise use kare koi tutorial hai", "general_question", False, ["hinglish"], "v4 hinglish"),
    ("mera payment ka refund kab tak aayega", "refund", True, ["hinglish"], "v4 hinglish"),
    ("aapne mera plan upgrade kyun nahi kiya", "account_issue", True, ["hinglish"], "v4 hinglish"),
    ("mujhe aapke yaha bulk order ka discount chahiye", "sales", False, ["hinglish"], "v4 hinglish"),
    ("aapki app ki design bahut achi hai", "other", False, ["hinglish"], "v4 hinglish"),
    ("mera account mein kisi aur ka data aa raha hai", "account_issue", True, ["hinglish"], "v4 hinglish"),
    ("aapne mujhe refund ka email kyun nahi bheja", "refund", False, ["hinglish"], "v4 hinglish"),
    ("mujhe aapke product ke baare mein aur jaanna hai", "general_question", False, ["hinglish"], "v4 hinglish"),
    ("aapki app ka latest version mein kya naya hai", "product_question", False, ["hinglish"], "v4 hinglish"),
    ("mera support ticket ka status kya hai", "general_question", False, ["hinglish"], "v4 hinglish"),
    ("aapne mujhe double charge kiya tha lekin ab refund kar diya", "refund", False, ["hinglish", "resolution_state"], "v4 hinglish"),
    ("maine problem report ki thi lekin ab solve ho gayi", "technical_support", False, ["hinglish", "resolution_state"], "v4 hinglish"),
    ("pehle aapki app slow thi but ab theek hai", "technical_support", False, ["hinglish", "resolution_state"], "v4 hinglish"),
    ("mujhe aapke yaha se monthly invoice chahiye", "pricing", False, ["hinglish"], "v4 hinglish"),
    ("aapki team bahut helpful hai thank you", "other", False, ["hinglish"], "v4 hinglish"),
    ("mera account delete karna hai par data backup chahiye", "general_question", False, ["hinglish"], "v4 hinglish"),
    ("aapke yaha koi referral bonus program hai", "general_question", False, ["hinglish"], "v4 hinglish"),
    ("mujhe aapke product ki latest pricing bhejo", "pricing", False, ["hinglish"], "v4 hinglish"),
    ("aapki app ka performance bahut slow hai", "technical_support", False, ["hinglish"], "v4 hinglish"),
    ("mera account ka security upgrade karna hai", "account_issue", False, ["hinglish"], "v4 hinglish"),
    ("mujhe aapke yaha bulk SMS feature chahiye", "product_question", False, ["hinglish"], "v4 hinglish"),
    ("aapne mujhe galat price quote kiya tha", "pricing", False, ["hinglish"], "v4 hinglish"),
    ("aapki team ka behavior bahut professional hai", "other", False, ["hinglish"], "v4 hinglish"),
    ("mera API key regenerate karo", "general_question", False, ["hinglish"], "v4 hinglish"),
    ("aapki app mein kuch options disable hain enable karo", "technical_support", False, ["hinglish"], "v4 hinglish"),
    ("mujhe aapke yaha par dedicated server chahiye", "sales", False, ["hinglish"], "v4 hinglish"),
])

# More noisy / real-world 
batch_add([
    ("need help with my accnt asap", "account_issue", True, ["noisy"], "v4 noisy"),
    ("y is the dashbord so confusin", "complaint", False, ["noisy"], "v4 noisy"),
    ("my paymnt got dedctd but no recpt", "pricing", False, ["noisy"], "v4 noisy"),
    ("the app is not respnding wen i tap", "technical_support", False, ["noisy"], "v4 noisy"),
    ("i cnt find the delet accnt buton", "general_question", False, ["noisy"], "v4 noisy"),
    ("ur chargin me rong amount since 3 months", "pricing", True, ["noisy"], "v4 noisy"),
    ("the search bar is not showin results", "technical_support", False, ["noisy"], "v4 noisy"),
    ("plz fix my account setup issue", "account_issue", False, ["noisy"], "v4 noisy"),
    ("i wnt to chnge my plan but cnt", "sales", False, ["noisy"], "v4 noisy"),
    ("ur app keeps crshing i m frustratd", "complaint", True, ["noisy"], "v4 noisy"),
    ("i dnt get y my invoice is so high", "pricing", False, ["noisy"], "v4 noisy"),
    ("need a human to call me urgent", "human_request", True, ["noisy"], "v4 noisy"),
    ("the feature i need is not availabl", "product_question", False, ["noisy"], "v4 noisy"),
    ("my team cant log in since ystrday", "account_issue", True, ["noisy"], "v4 noisy"),
    ("u guys need to fix ur app its slow", "technical_support", False, ["noisy"], "v4 noisy"),
    ("i wnt my subscripn canld immeditly", "sales", False, ["noisy"], "v4 noisy"),
    ("refnd my muney u scammers", "refund", True, ["noisy"], "v4 noisy"),
    ("hw do i use the template builder", "product_question", False, ["noisy"], "v4 noisy"),
    ("the updat brok my workflow", "technical_support", True, ["noisy"], "v4 noisy"),
    ("i m lockd out of my admn account", "account_issue", True, ["noisy"], "v4 noisy"),
    ("ur team is useless never reply", "complaint", True, ["noisy"], "v4 noisy"),
    ("hw can i get a refund for my plan", "refund", False, ["noisy"], "v4 noisy"),
    ("the dashbord loadz forever", "technical_support", False, ["noisy"], "v4 noisy"),
    ("i need 2 talk to a manager", "human_request", True, ["noisy"], "v4 noisy"),
    ("my notifcatns are not workin", "technical_support", False, ["noisy"], "v4 noisy"),
    ("the integratn is broken fix it", "technical_support", False, ["noisy"], "v4 noisy"),
    ("y is the pricin so confusing", "pricing", False, ["noisy"], "v4 noisy"),
    ("i cnt chang my email in settngs", "account_issue", False, ["noisy"], "v4 noisy"),
    ("ur websit is nt loadin at all", "technical_support", False, ["noisy"], "v4 noisy"),
    ("i m very angy abt the service", "complaint", False, ["noisy"], "v4 noisy"),
    ("pls process my refund request fast", "refund", True, ["noisy"], "v4 noisy"),
    ("how does the pricin werk for api", "pricing", False, ["noisy"], "v4 noisy"),
    ("i cnt figur the new updat layout", "general_question", False, ["noisy"], "v4 noisy"),
    ("my report genrator is brokn", "technical_support", False, ["noisy"], "v4 noisy"),
    ("need help setting up my workspace", "general_question", False, ["noisy"], "v4 noisy"),
    ("ur bot is dumm i need reel person", "human_request", True, ["noisy"], "v4 noisy"),
    ("the calender sync is not wrking", "technical_support", False, ["noisy"], "v4 noisy"),
    ("i wnt to knw the price of premium", "pricing", False, ["noisy"], "v4 noisy"),
    ("my team membr cant join the work", "account_issue", False, ["noisy"], "v4 noisy"),
    ("u guys charged me twice this month", "pricing", True, ["noisy"], "v4 noisy"),
])

# More escalation negatives with the don't-call pattern
batch_add([
    ("Please do not call me. Just reply via email.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("I do not wish to speak to anyone. Text only.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("Don't contact me. I'll follow up if needed.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("I expressly forbid any phone calls about this.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("No calls, no emails, no follow up. Got it?", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("Do not escalate this to anyone. Understand?", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("I don't need anyone to contact me. Period.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("Please respect my request and do not call.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
    ("Not a request for help. Just giving feedback.", "complaint", False, ["hard_negative_escalation"], "v4 esc-"),
    ("I don't consent to being contacted about this.", "general_question", False, ["hard_negative_escalation"], "v4 esc-"),
])

# More escalation positives with business impact
batch_add([
    ("I'm losing revenue every minute this is down.", "technical_support", True, ["escalation_positive"], "v4 esc+"),
    ("This is a critical system that needs immediate restoration.", "technical_support", True, ["escalation_positive"], "v4 esc+"),
    ("My team cannot function without this feature.", "technical_support", True, ["escalation_positive"], "v4 esc+"),
    ("I need written confirmation of the resolution timeline.", "complaint", True, ["escalation_positive"], "v4 esc+"),
    ("This issue has been escalated to my legal team.", "complaint", True, ["escalation_positive"], "v4 esc+"),
    ("I expect a call from your VP of Customer Experience.", "human_request", True, ["escalation_positive"], "v4 esc+"),
    ("There are regulatory implications if this is not fixed.", "technical_support", True, ["escalation_positive"], "v4 esc+"),
    ("My customer data is at risk because of this bug.", "technical_support", True, ["escalation_positive"], "v4 esc+"),
    ("I need a guaranteed resolution by tomorrow morning.", "technical_support", True, ["escalation_positive"], "v4 esc+"),
    ("Your incompetence is costing my business real money.", "complaint", True, ["escalation_positive"], "v4 esc+"),
])

# More negation
batch_add([
    ("Not a refund request. Just asking about policy.", "general_question", False, ["negation"], "v4 negation"),
    ("I am not complaining. Just providing feedback.", "other", False, ["negation"], "v4 negation"),
    ("I didn't ask for a call. I asked for an email.", "general_question", False, ["negation"], "v4 negation"),
    ("Don't cancel anything. I was just checking.", "general_question", False, ["negation"], "v4 negation"),
    ("I never requested a refund. Read my message again.", "general_question", False, ["negation"], "v4 negation"),
    ("Not a support ticket. This is a suggestion.", "other", False, ["negation"], "v4 negation"),
    ("Do not process any changes to my account.", "general_question", False, ["negation"], "v4 negation"),
    ("I said don't escalate. Please listen.", "general_question", False, ["negation"], "v4 negation"),
    ("I don't want compensation. I want an explanation.", "general_question", False, ["negation"], "v4 negation"),
    ("No need to take any action. Just informing.", "general_question", False, ["negation"], "v4 negation"),
])

# ============================================================
# SHUFFLE AND FINALIZE
# ============================================================
random.shuffle(examples)

print(f"Total examples: {len(examples)}")
print(f"Carried over from v1: {v1_count}")
print(f"New examples added: {len(examples) - v1_count}")

intent_counts = {}
for ex in examples:
    intent_counts[ex["intent"]] = intent_counts.get(ex["intent"], 0) + 1
print(f"Intent distribution: {json.dumps(intent_counts, indent=2)}")

esc_counts = {"true": 0, "false": 0}
for ex in examples:
    esc_counts[ex["escalation_required"]] += 1
print(f"Escalation distribution: {json.dumps(esc_counts, indent=2)}")

tag_counts = {}
for ex in examples:
    tags = json.loads(ex["tags"])
    for t in tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
print(f"Tag distribution: {json.dumps(tag_counts, indent=2)}")

# Write CSV
output_csv = os.path.join(V3_DIR, "dataset_v3_raw.csv")
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "text", "intent", "escalation_required", "language_style",
        "difficulty", "scenario_type", "notes", "tags"
    ])
    writer.writeheader()
    writer.writerows(examples)

print(f"Dataset written to {output_csv}")

# Write info JSON
info = {
    "dataset_version": "dataset-v3",
    "total_examples": len(examples),
    "v1_examples_carried_over": v1_count,
    "new_examples_added": len(examples) - v1_count,
    "random_seed": 42,
    "intents": sorted(INTENTS),
    "intent_distribution": intent_counts,
    "escalation_distribution": esc_counts,
    "tag_distribution": tag_counts,
}
output_info = os.path.join(V3_DIR, "dataset_v3_info.json")
with open(output_info, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"Info written to {output_info}")
print("Done! Dataset-v3 ready for validation/split pipeline.")
