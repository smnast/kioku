# Kioku

<details>
    <summary>DQN</summary>
    <img src="results/dqn_cartpole/dqn_cartpole.png" alt="DQN CartPole episode rewards" />
    <h3>Using:</h3>
    <ul>
        <li>DQN</li>
        <li>Double q networks</li>
        <li>Epsilon decay</li>
    </ul>
    <p><b>Masters cartpole after only 92 episodes?</b></p>
    <img src="results/dqn_cartpole/dqn_cartpole.gif" alt="DQN CartPole gif" />
</details>

<details>
    <summary>A2C</summary>
    <img src="results/a2c_cartpole/a2c_cartpole.png" alt="A2C CartPole episode rewards" />
    <h3>Using:</h3>
    <ul>
        <li>A2C</li>
        <li><s>GAE</s></li>
        <li>N-Step Returns <s>(with GAE)</s></li>
    </ul>
    <p>
        Note that A2C is <i>much</i> less sample efficient than DQN
        and the SOTA (PPO, TD3, SAC).
    <p>
    <h3>Lunar Lander:</h3>
    <img src="results/a2c_lunarlander/a2c_lunarlander.png" alt="A2C Lunar Lander episode rewards" />
    <img src="results/a2c_lunarlander/a2c_lunarlander.gif" alt="A2C Lunar Lander gif" />
</details>

<details>
    <summary>PPO</summary>
    <img src="results/ppo_cartpole/ppo_cartpole.png" alt="PPO CartPole episode rewards" />
    <p><b>...did I just successfully implement PPO?</b></p>
    <h3>Using:</h3>
    <ul>
        <li>PPO (basically A2C with a few extra steps)</li>
        <li>GAE</li>
        <li>N-Step Returns (with GAE)</li>
        <li>Mini-batch learning</li>
        <li>Multiple learning iterations per batch</li>
    </ul>
    <p><b>As sample efficient as DQN?</b> (minus the memory)</p>
    <img src="results/ppo_cartpole/ppo_cartpole.gif" alt="PPO CartPole gif" />
    <h3>Lunar Lander:</h3>
    <img src="results/ppo_lunarlander/ppo_lunarlander.png" alt="PPO Lunar Lander episode rewards" />
    <img src="results/ppo_lunarlander/ppo_lunarlander.gif" alt="PPO Lunar Lander gif" />
</details>