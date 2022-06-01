# Deep Q-learning Trading Bot
Reinforcement learning trading bot that can buy, sell and hold stocks using deep Q-Learning

## 1. Modeling the trading problem

1. States $S_{t}$:
   - Number of shares owned at $t$
   - Stock price at $t$
   - Amount of cash available at $t$
        
2. Agent's action space at every discrete $t$:
   - Buy until no cash left
   - Hold to earn risk free interest rate
   - Liquidize all stock

        
3. Environment
   - Provides states $S_{t}$ and $S_{t+1}$
   - Provide daily return of bot as reward


![runs](https://user-images.githubusercontent.com/88515336/171457932-e464389c-118f-4b70-967e-6bae8074aea6.png)
![run](https://user-images.githubusercontent.com/88515336/171457986-9a62df79-241a-4842-99b4-c4204c352377.png)

