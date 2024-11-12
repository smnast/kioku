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
    <img src="results/a2c_lunarlander/a2c_lunarlander.png">
    <img src="results/a2c_lunarlander/a2c_lunarlander.gif">
</details>
