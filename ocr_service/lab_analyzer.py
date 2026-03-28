from groq import Groq
import os
from dotenv import load_dotenv
import re

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def numbers(s):
    return [float(x) for x in re.findall(r'\d+\.?\d*', s)]

def find_range(nums):
    if len(nums) < 3:
        return None

    value = nums[0]
    for i in range(1,len(nums)):
        for j in range(i+1,len(nums)):
            low=min(nums[i],nums[j])
            high=max(nums[i],nums[j])
            if low < value < high:
                return value,low,high
    return None


def analyze_lab_report(text):

    lines=[l.strip() for l in text.split("\n") if l.strip()]
    findings=[]
    normal_tests=[]

    for i,line in enumerate(lines):

        if not re.search(r'[a-zA-Z]',line):
            continue

        context=line
        for k in range(1,4):
            if i+k<len(lines):
                context+=" "+lines[i+k]

        nums=numbers(context)
        result=find_range(nums)

        if result:
            value,low,high=result
            test=line[:25]

            if value<low:
                findings.append(f"{test} LOW ({value})")
            elif value>high:
                findings.append(f"{test} HIGH ({value})")
            else:
                normal_tests.append(test)

    # ---------- LLM EXPLANATION ----------
    structured_data = f"""
Normal Tests:
{", ".join(normal_tests)}

Abnormal Tests:
{", ".join(findings) if findings else "None"}
"""

    prompt=f"""
You are a doctor explaining a lab report to a patient in simple words.

Based ONLY on the structured results below,
explain health condition.

Do NOT invent diseases.
If normal → reassure patient.

Report Data:
{structured_data}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )

    return res.choices[0].message.content