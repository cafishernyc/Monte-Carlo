import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_paths(
    start_value=0,
    current_age=55,
    retirement_age=60,
    end_age=90,
    n_sims=20_000,
    exp_return=0.06,
    volatility=0.12,
    annual_contribution=0,        # net contributions until retirement
    annual_spending=0,      # withdrawals starting at retirement (inflation-adjusted)
    inflation=0.03,
    model="Lognormal",
    seed=42,
    post_retirement_contribution=0, # New parameter: contributions after retirement
    post_retirement_contribution_end_age=90, # New parameter
):
    years = max(1, end_age - current_age)
    retire_year = max(0, min(years, retirement_age - current_age))

    rng = np.random.default_rng(seed)

    if model == "Lognormal":
        mu = np.log(1 + exp_return) - 0.5 * volatility**2
        rets = np.exp(rng.normal(mu, volatility, size=(n_sims, years))) - 1.0
    else:
        rets = rng.normal(exp_return, volatility, size=(n_sims, years))
        rets = np.clip(rets, -0.95, None)

    paths = np.empty((n_sims, years + 1), dtype=np.float64)
    paths[:, 0] = start_value

    ruined = np.zeros(n_sims, dtype=bool)
    ruin_by_year = np.zeros(years + 1, dtype=np.float64)
    ruin_by_year[0] = 0.0

    for t in range(1, years + 1):
        prev = paths[:, t - 1]
        new = prev * (1.0 + rets[:, t - 1])

        # Contributions before retirement (net income added at end of year)
        if t <= retire_year and annual_contribution > 0:
            new = new + annual_contribution * ((1.0 + inflation) ** (t - 1))

        # Withdrawals starting at retirement (inflation-adjusted)
        # AND Post-retirement contributions
        if t > retire_year:
            if annual_spending > 0:
                w = annual_spending * ((1.0 + inflation) ** (t - retire_year - 1))
                new = new - w
            if post_retirement_contribution > 0 and (current_age + t) <= post_retirement_contribution_end_age:
                p = post_retirement_contribution * ((1.0 + inflation) ** (t - retire_year - 1))
                new = new + p

        ruin_now = new <= 0
        ruined |= ruin_now
        new = np.where(new > 0, new, 0.0)

        paths[:, t] = new
        ruin_by_year[t] = ruined.mean()

    pct_levels = [10, 25, 50, 75, 90]
    bands = {p: np.percentile(paths, p, axis=0) for p in pct_levels}
    terminal = paths[:, -1]
    # Calculate probability of success instead of ruin
    prob_success = 1.0 - ruined.mean() if annual_spending > 0 else np.nan

    return years, retire_year, paths, bands, terminal, ruin_by_year, prob_success

def fmt_money(x):
    if x >= 1e9:  return f"${x/1e9:,.2f}B"
    if x >= 1e6:  return f"${x/1e6:,.2f}M"
    return f"${x:,.0f}"


st.set_page_config(layout="wide")
st.title("Financial Monte Carlo Simulation")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Replaced st.sidebar.slider with st.sidebar.number_input
start_value = st.sidebar.number_input("Starting Portfolio Value", min_value=0, max_value=100_000_000, value=0, step=100_000)
current_age = st.sidebar.slider("Current Age", 25, 80, 50, step=1)
retirement_age = st.sidebar.slider("Retirement Age", 30, 85, 60, step=1)
end_age = st.sidebar.slider("End Age", 50, 100, 90, step=1)

annual_contribution = st.sidebar.slider("Annual Contribution until Retirement", 0, 2_000_000, 0, step=25_000)
annual_spending = st.sidebar.slider("Annual Spending after Retirement", 0, 2_000_000, 0, step=25_000)
post_retirement_contribution = st.sidebar.slider("Annual Contribution after Retirement", 0, 2_000_000, 0, step=25_000)
post_retirement_contribution_end_age = st.sidebar.slider("End Age for Post-Retirement Contributions", retirement_age, end_age, end_age, step=1)

exp_return = st.sidebar.slider("Expected Annual Return", 0.00, 0.12, 0.06, step=0.0025, format="%.2f%%")
volatility = st.sidebar.slider("Annual Volatility", 0.02, 0.30, 0.12, step=0.005, format="%.2f%%")
inflation = st.sidebar.slider("Annual Inflation Rate", 0.00, 0.06, 0.03, step=0.0025, format="%.2f%%")

n_sims = st.sidebar.slider("Number of Simulations", 2_000, 80_000, 20_000, step=2_000)
model = st.sidebar.selectbox("Return Model", ["Lognormal", "Normal (clipped)"], index=0)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)


# --- Input validation ---
if retirement_age < current_age:
    st.sidebar.error("Retirement age cannot be less than current age.")
    st.stop()
if end_age < retirement_age:
    st.sidebar.error("End age cannot be less than retirement age.")
    st.stop()
if post_retirement_contribution_end_age < retirement_age:
    st.sidebar.error("Post-retirement contribution end age cannot be less than retirement age.")
    st.stop()


# Run simulation and display results
if st.sidebar.button("Run Simulation"):
    years, retire_year, paths, bands, terminal, ruin_by_year, prob_success = simulate_paths(
        start_value=start_value,
        current_age=current_age,
        retirement_age=retirement_age,
        end_age=end_age,
        n_sims=n_sims,
        exp_return=exp_return,
        volatility=volatility,
        annual_contribution=annual_contribution,
        annual_spending=annual_spending,
        inflation=inflation,
        model=model,
        seed=seed,
        post_retirement_contribution=post_retirement_contribution,
        post_retirement_contribution_end_age=post_retirement_contribution_end_age
    )

    t = np.arange(years + 1)
    ages = current_age + t
    retire_age_eff = current_age + retire_year

    st.header("Simulation Results")

    # Chart 1: Probability bands over time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.fill_between(ages, bands[10], bands[90], alpha=0.25, label="10–90% band")
    ax1.fill_between(ages, bands[25], bands[75], alpha=0.35, label="25–75% band")
    ax1.plot(ages, bands[50], linewidth=2, label="Median path")
    ax1.axvline(retire_age_eff, linestyle="--", linewidth=2, color='red', label=f"Retire @ {retire_age_eff}")
    ax1.set_title("Portfolio Value Over Time — Probability Bands")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Portfolio Value (Nominal $)")
    ax1.legend()
    st.pyplot(fig1)

    # Chart 2: Terminal distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(terminal, bins=60)
    ax2.set_title(f"Ending Value Distribution (Age {end_age})")
    ax2.set_xlabel("Ending Value (Nominal $)")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # Chart 3: Probability of success over time (if spending)
    if annual_spending > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        # Invert ruin_by_year to show probability of success
        prob_success_by_year = 1.0 - ruin_by_year
        ax3.plot(ages, prob_success_by_year, linewidth=2)
        ax3.axvline(retire_age_eff, linestyle="--", linewidth=2, color='red')
        ax3.set_title("Probability of Success Over Time")
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Probability")
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)

    # Summary stats
    st.subheader("Summary Statistics")
    st.write(f"**Inputs:**")
    st.write(f"  Start value: {fmt_money(start_value)}")
    st.write(f"  Ages: {current_age} → {end_age} (years={years}), Retire @ {retire_age_eff}")
    st.write(f"  Sims: {n_sims:,} | Return model: {model}")
    st.write(f"  Expected return: {exp_return*100:.2f}% | Volatility: {volatility*100:.2f}% | Inflation: {inflation*100:.2f}%")
    st.write(f"  Annual contribution until retirement: {fmt_money(annual_contribution)}/yr")
    st.write(f"  Annual spending after retirement: {fmt_money(annual_spending)}/yr")
    st.write(f"  Annual contribution after retirement: {fmt_money(post_retirement_contribution)}/yr (until age {post_retirement_contribution_end_age})")
    if annual_spending > 0:
        st.write(f"  Probability of success: {prob_success*100:.2f}%")

    p10, p25, p50, p75, p90 = (np.percentile(terminal, q) for q in (10, 25, 50, 75, 90))
    st.write(f"\n**Ending Value (Nominal) Percentiles:**")
    st.write(f"  P10: {fmt_money(p10)}")
    st.write(f"  P25: {fmt_money(p25)}")
    st.write(f"  P50: {fmt_money(p50)}")
    st.write(f"  P75: {fmt_money(p75)}")
    st.write(f"  P90: {fmt_money(p90)}")

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")
