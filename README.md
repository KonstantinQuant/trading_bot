# Deep Q-learning Trading Bot
Reinforcement learning trading bot that can buy, sell and hold stocks using deep Q-Learning

![runs](https://user-images.githubusercontent.com/88515336/171457932-e464389c-118f-4b70-967e-6bae8074aea6.png)
![run](https://user-images.githubusercontent.com/88515336/171457986-9a62df79-241a-4842-99b4-c4204c352377.png)

## Modeling the trading problem
$
\begin{itemize}
        \item States $S_{t}$:
        \begin{enumerate}
            \item Number of shares owned at $t$
            \item Stock price at $t$
            \item Amount of cash available at $t$
        \end{enumerate}
        
        \item Agent's action space at every discrete $t$:
        \begin{enumerate}
            \item Buy until no cash left
            \item Hold to earn risk free interest rate
            \item Liquidize all stock
        \end{enumerate}
        
        \item Environment
        \begin{itemize}
            \item Provides states $S_{t}$ and $S_{t+1}$
            \item Provide daily return of bot as reward
        \end{itemize}
\end{itemize}
$
