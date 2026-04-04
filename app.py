from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd

app = Flask(__name__)
app.secret_key = "churn_secret"

# ================= LOAD DATA =================
df = pd.read_csv("prediction_output.csv")
df.columns = df.columns.str.strip()

# ⭐ NEW RISK LOGIC (UPDATED AS YOU ASKED)
def risk_category(p):
    if p >= 65:
        return "High"
    elif p >= 30:
        return "Medium"
    else:
        return "Low"

df["Risk"] = df["Churn_Prob"].apply(risk_category)

# ================= LOGIN =================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "bank123":
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    total = len(df)
    high = len(df[df["Risk"]=="High"])
    medium = len(df[df["Risk"]=="Medium"])
    low = len(df[df["Risk"]=="Low"])

    return render_template(
        "dashboard.html",
        total=total,
        high=high,
        medium=medium,
        low=low
    )


# ================= CUSTOMER LIST =================
@app.route("/customers")
def customers():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "customer.html",     # ⭐ IMPORTANT (your template name)
        data=df.to_dict(orient="records")
    )


# ================= HIGH RISK PAGE =================
@app.route("/highrisk")
def highrisk():
    if "user" not in session:
        return redirect(url_for("login"))

    high_df = df[df["Risk"]=="High"]

    return render_template(
        "highrisk.html",
        data=high_df.to_dict(orient="records")
    )


# ================= CUSTOMER DETAIL =================
@app.route("/customer/<cid>")
def customer(cid):
    if "user" not in session:
        return redirect(url_for("login"))

    row = df[df["CustomerId"] == int(cid)]

    if row.empty:
        return "Customer Not Found"

    data = row.iloc[0].to_dict()

    return render_template(
        "customer_detail.html",
        c=data
    )


# ================= RETENTION STRATEGY =================
@app.route("/retention/<cid>")
def retention(cid):
    if "user" not in session:
        return redirect(url_for("login"))

    row = df[df["CustomerId"] == int(cid)]
    prob = float(row["Churn_Prob"].values[0])

    if prob >= 80:
        strategies = [
            "Premium Credit Card Upgrade",
            "Dedicated Relationship Manager",
            "Loan Interest Discount",
            "10% Cashback Offers",
            "Free Insurance Bundle"
        ]
    elif prob >= 65:
        strategies = [
            "Loyalty Reward Points Boost",
            "Special Fixed Deposit Interest",
            "Mobile Banking Privileges",
            "Personalised Banking Offers"
        ]
    else:
        strategies = [
            "Financial Advisory Emails",
            "Customer Engagement Campaign"
        ]

    return render_template(
        "retention.html",
        s=strategies,
        cid=cid
    )


# ================= DOWNLOAD CSV =================
@app.route("/download")
def download():
    return send_file("prediction_output.csv", as_attachment=True)


# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
