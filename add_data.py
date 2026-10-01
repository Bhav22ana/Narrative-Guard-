import csv

file_path = "dataset/annotated/narrativeguard_dataset.csv"

conversation = """Customer: Hello sir, enakku personal loan options pathi therinjukanum.

Bank Representative: Sure sir. Loan amount and tenure based on your eligibility, different options irukkum.

Customer: Enakku around ₹2 lakh loan venum. Interest rate and processing fee details sollunga.

Bank Representative: Interest rate and applicable fees loan type and eligibility based on vary aagum. Official loan documents-la complete details irukkum.

Customer: Loan apply panna first payment edhavadhu panna venduma?

Bank Representative: Illa sir. Loan application submit panna upfront payment required illa. Official application process moolama apply pannalam.

Customer: Enna documents venum?

Bank Representative: Usually identity and income-related documents required irukkalam. Exact requirements official application process-la display aagum.

Customer: Okay sir. Naan official bank website and application-la details check pannitu apply pannikiren.

Bank Representative: Sure sir. Loan terms, interest rate, fees and repayment conditions ellam review pannitu decision edunga."""

row = {
    "conversation_id": "CONV_0020",
    "language": "Tanglish",
    "source_type": "synthetic",
    "source_reference": "NarrativeGuard Pilot Dataset",
    "scam_type": "genuine_financial_conversation",
    "conversation": conversation,
    "claim": "Customer is asking about legitimate personal loan options, interest rates, fees and application requirements.",
    "target_action": "Review official loan terms and apply through the official bank application if suitable.",
    "manipulation_tactics": "none",
    "evidence_required": "official_loan_terms;interest_rate;fee_information;eligibility_requirements",
    "evidence_available": "Yes",
    "verification_status": "verified",
    "risk_level": "low",
    "recommended_action": "Review official loan terms, fees, interest rate and repayment conditions before applying through the official channel.",
    "explanation": "The conversation is a normal loan inquiry. The representative does not demand an upfront payment or sensitive authentication information, and the customer chooses to verify the information through official channels."
}
with open(file_path, "a", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=row.keys())
    writer.writerow(row)

print("CONV_0020 added successfully!")