# Roostoo Quant Trading Hackathon — Official FAQ

# 📌 1. Registration & Confirmation

### Q1. I registered but didn’t receive a confirmation email. Am I registered?

**Ans:** If you filled in the official Typeform (or your team captain filled it for you), your team is confirmed. Confirmation/resource emails will be sent later on or before 3/16. 

### Q2. Do all team members need to register individually?

**Ans:** No. If the team captain includes all members in the Typeform submission, additional registrations are not required.

### Q3. I mistyped my email during registration. What should I do?

**Ans:** DM the organizer with:

- Correct email
- Team name.

We will update it.

### Q4. Can I add/update my team members?

**Ans:** Yes, you can add more teammate later on by resubmitting the TypeForm registration with same captain + teamName, before the 3/14 deadline. 

### Q5. Can I still join after missing the info session?

**Ans:** Yes. The recording will be shared in the group or via email.

---

# 📌 2. Info Session & Resources

### Q6. Where can I find the recording of the info session?

**Ans:** It will be shared via Google Drive link in the WhatsApp group.

### Q7. Where can I access competition documentation and resources?

**Ans:** In the resource pack email and Roostoo GitHub API documentation.

---

# 📌 3. AWS Setup & Server Issues

### Q8. I received AWS setup email but no invitation link.

**Ans:** The invitation link may arrive later. Please wait or contact organizers.

### Q9. I cannot find “HackathonInstanceRole” when launching EC2.

**Ans:**

- Make sure you signed in using the **invitation email link**
- The role rollout may take time
- If still not visible, contact organizers with team number + email

### Q10. I can’t log into AWS even after resetting password.

**Ans:**

- Use the correct invitation link
- Set a new password
- Complete MFA setup
- Follow instructions in the Notion guide

### Q11. After launching EC2, I see errors.

**Ans:** Wait a few minutes and refresh. Instance provisioning may take time.

### Q12. Can I change EC2 storage volume size?

**Ans:** You cannot modify the volume if permissions are restricted.

Solution:

Delete instance → Launch a new one with larger volume.

### Q13. Are we limited to Sydney region?

**Ans:** Yes (unless stated otherwise). Changing region may not be allowed.

---

# 📌 4. Market Data Usage

### Q14. Can we use other APIs for market data?

**Ans:** Yes.

### Q15. Is Roostoo real-time data different from Binance?

**Ans:** No. Roostoo real-time pricing is streamed from Binance.

### Q16. Does Roostoo API provide OHLCV?

**Ans:** No. It provides ticker snapshot.

For OHLCV, use external data providers in our data source pack resources. 

---

# 📌 5. Roostoo API Usage

### Q17. How do I authenticate with Roostoo API?

**Ans:**

- Use API Key
- Sign payload using HMAC SHA256
- Include timestamp
- Send `RST-API-KEY` and `MSG-SIGNATURE` headers

### Q18. What are the main API endpoints?

- `/v3/serverTime`
- `/v3/exchangeInfo`
- `/v3/ticker`
- `/v3/balance`
- `/v3/place_order`
- `/v3/query_order`
- `/v3/cancel_order`

---

### Q19. I get “HTTP: Max retries exceeded” error. What should I do?

**Ans:**

- Implement retry with exponential backoff
- Reduce request frequency
- Catch exceptions properly

### Q20. Is there a rate limit?

**Ans:** Yes, 30 calls per minute.

Calls more than limit → failed responses. Also, high trading frequency eats away your commission costs and returns so you’ll want to avoid it. 

### Q21. What is the upper bound on trades per minute?

**Ans:** Yes, 30 calls including query, execute. Keep trading frequency reasonable to avoid API failures.

### Q22. Why do I see small negative balances like -0.01 after selling?

**Ans:** Rounding error. Harmless.

---

# 📌 6. Competition Rules

### Q23. When does the competition start?

**Ans:** 1st round : Mar 21, 8:00 PM

  2nd round : Apr 4, 8:00 PM

### Q24. When must the first trade execute?

**Ans:**  1st round : Before Mar 22, 8:00 PM

   2nd round : Before Apr 5, 8:00 PM

### Q25. Will we be disqualified if no trades on first day?

**Ans:** Not disqualified, but finalist and prize consideration favors bots active for full period.

### Q26. Can we modify our bot after competition starts?

**Ans:**

- Yes, you are allowed to update or improve strategies during the competition period to adapt to changing market conditions.
- However, the following rules apply:
    - All strategy or code changes must be committed and recorded in your repository. Each update must have a clear commit history so we can track modifications and ensure fair participation.
    - Manual intervention is strictly prohibited. You may not manually stop the bot, override its decisions, or place discretionary trades through the API.
    - All trades must be generated autonomously by your submitted bot logic.

### Q27. Do we need to liquidate holdings at the end?

**Ans:** System will automatically liquidate.

Still recommended to pause bots to save resources.

### Q28. Which API keys should we use for competition?

**Ans:** You’ll receive two set of API keys. 

- One for testing purpose on your Roostoo general account portfolio.
- One for the **official** **first round competition. Please don’t mis-use API keys.**

If you are selected for the **Finalist Round**, you will receive the third set – a new and separate set of API keys specifically for that round.

---

# 📌 7. Monitoring & Leaderboard

### Q29. How do we monitor portfolio during testing?

**Ans:** Use API:

- `/v3/balance`
- `/v3/query_order`

Frontend dashboard not available during testing.

### Q30. How do we monitor during competition?

**Ans:** Through Roostoo App leaderboard:

- Orders
- Portfolio returns
- Ranking (real-time)

### Q31. Will we get frontend trading access?

**Ans:** No.

Quant competition = API only (no manual trades).

---

# 📌 8. Submission Requirements

### Q32. How do we submit final work?

**Ans:**

- Submit GitHub repository (must be open-source)
- Running bot on leaderboard acts as proof of deployment

### Q33. Is AWS only used for deployment?

**Ans:** Yes. It hosts your live trading bot.

---

# 📌 9. Bot Development & Deployment

### Q34. Can we backtest before competition?

**Ans:** Yes using:

- Binance historical data
- Other public APIs

### Q35. Do we have access to historical data before competition?

**Ans:** Yes via external sources.

### Q36. Should bots be autonomous?

**Ans:** Yes. Must trade automatically without manual intervention.

---

# 📌 11. Best Practices

### Q37. What’s best practice for repository management?

Recommended structure:

```
bot/
  strategy/
  execution/
  data/
  config/
  logs/
tests/
requirements.txt
Dockerfile
README.md
```

Best practices:

- Use `.env` for API keys
- Add clear README
- Use Git branches (main/dev)
- Tag final submission version
- Keep it reproducible

### Q38. Should teams log their trades?

Yes — strongly recommended.

Minimum logging:

- Timestamp
- Symbol
- Side
- Price
- Quantity
- Order ID
- API response

Optional:

- PnL
- Signal reason
- Strategy state

### Q39. How should bots log activity?

Options:

- Local file logging
- CloudWatch logs
- Database (SQLite/Postgres)
- CSV export for evaluation

### Q40. What to do if Binance API blocks US region?

- Use Roostoo API (same pricing stream)
- Use Binance data dump site
- Use proxy only if allowed
- Confirm allowed regions before competition

### Q41. What should be included in the README file of repository?

Ans: Your README should clearly explain how your bot works and how to run it. Judges should be able to understand and reproduce your project easily.

Recommended README Structure (feel free to create your own structure too):

1. **Project** **Overview**
    - Short description of your strategy
    - High-level idea (e.g., momentum, mean reversion, ML-based, arbitrage, etc.)
    - Key features
2. **Architecture**
    - System design diagram (optional but recommended)
    - Components (data module, strategy module, execution module, logging module)
    - Tech stack used
3. **Strategy Explanation**
    - Entry conditions
    - Exit conditions
    - Risk management rules
    - Position sizing logic
    - Any assumptions made
4. **Setup instructions & How to run bot**

---

# 📌 12. Common Technical Errors & Solutions

| Issue | Likely Cause | Fix |
| --- | --- | --- |
| 422 Error | Invalid params | Check API docs |
| Max retries exceeded | Too many requests | Reduce frequency |
| Negative balance -0.01 | Rounding | Ignore |
| Cannot see IAM role | Wrong login method | Use invitation email |
| Instance error | Not fully provisioned | Wait + refresh |

---