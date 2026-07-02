import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CARL.envs.TwoArmedBandit import TwoArmedBandit
from CARL.agents.SimpleConf import SimpleConf

# Task parameters
p_0 = 0.6
p_2 = 0.4
reward = 1
punishment = -1
n_simulations = 500
n_agents = 10 #number of agents in the group
n_trials = 200
reversal_trials = [50,100,150]  # reversals
beta = 4  # Fixed beta for this sweep

# Alpha sweep values
alpha_vals = np.round(np.arange(0.05, 1.0, 0.1), 2)
heatmap_data = np.zeros((len(alpha_vals), len(alpha_vals)))  # rows: alphaC, cols: alphaD

# Data storage
all_records = []

def run_multi_agent_sim(alphaC, alphaD, beta):
    avg_payoff = 0
    for sim in range(n_simulations):
        # Same parameters for all agents
        pars_agent = np.tile([alphaC / n_agents, alphaD / n_agents, beta], (n_agents, 1))
        agent = SimpleConf(pars_agent)
        bandit = TwoArmedBandit(p_0, p_2, reward, punishment)
        Qtable = np.zeros((n_agents, 2))
        G = agent.connect_agents_full()

        sim_payoff = 0
        for t in range(n_trials):
            if t in reversal_trials:
                bandit.reversal_occurs()

            choices = agent.all_take_action(Qtable)
            payoffs = bandit.return_payoffs(choices)
            Qtable = agent.update_Qvalues(G, choices, payoffs, Qtable)
            sim_payoff += np.mean(payoffs)  # average payoff across agents for this trial
            # Save per trial data
            for a in range(n_agents):
                all_records.append({
                    "simulation": sim,
                    "trial": t,
                    "agent": a,
                    "alphaC": alphaC,
                    "alphaD": alphaD,
                    "beta": beta,
                    "choice": choices[a],
                    "payoff": payoffs[a],
                    "Q0": Qtable[a, 0],
                    "Q1": Qtable[a, 1]
                })
        avg_payoff += sim_payoff / n_trials  # average payoff per trial

    return avg_payoff / n_simulations

# Run sweep
for i, alphaC in enumerate(alpha_vals):
    for j, alphaD in enumerate(alpha_vals):
        print(f"Running alphaC={alphaC}, alphaD={alphaD}")
        avg_trial_payoff = run_multi_agent_sim(alphaC, alphaD, beta)
        heatmap_data[i, j] = avg_trial_payoff

# Save all detailed data to CSV
df = pd.DataFrame(all_records)
df.to_csv("HM_Mixed_3R_10gents_new.csv", index=False)

# Plot heatmap using matplotlib only
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(heatmap_data, cmap="viridis", origin="lower", aspect="auto")

# Add text annotations
#for i in range(len(alpha_vals)):
    #for j in range(len(alpha_vals)):
        #ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                #ha='center', va='center',
                #color='white' if heatmap_data[i, j] < 0.5 else 'black')

# Set ticks and labels
ax.set_xticks(np.arange(len(alpha_vals)))
ax.set_yticks(np.arange(len(alpha_vals)))
ax.set_xticklabels(alpha_vals)
ax.set_yticklabels(alpha_vals)
ax.set_xlabel("alphaD")
ax.set_ylabel("alphaC")
ax.set_title(f"Average Payoff per Trial per Agent (β={beta}, n_agents={n_agents})")

# Rotate x-axis labels for clarity
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar
fig.colorbar(cax)
plt.tight_layout()
plt.show()
fig.savefig('HM_Mixed_3R_10gents_new.pdf')
