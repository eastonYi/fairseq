# Experiments

## wav2vec 2.0 egs
\begin{table*}[ht]
    \centering
    \caption{Performance on CALLHOME 6 language corpus. The criteria is WER(\%) on each test set.}
    \begin{tabular}{ l l l l l l l }
    \toprule
    \textbf{Models} & AR & EN & MA & JA & GE & SP \\
    \midrule
    mlstm-residual \cite{zhou2017multilingual} & 56.47 & 43.93 & 45.85 & 50.13 & 51.75 & 53.38  \\
    \midrule
     \tabincell{l}{Speech-Transformer \cite{zhou2018multilingual} \\ \quad + HKUST pre-training } & 48.35 & 33.77 & 37.62 & 36.99 & 44.98 & 51.54  \\
    \bottomrule
    \tabincell{l}{wav2vec2 encoder small \\ \quad  + ctc (letter) + LM}
                    & 45.53 & 24.05 & 43.05 & 38.40 & 41.83 & 50.41 \\
    \midrule
    \tabincell{l}{wav2vec2 encoder small \\ \quad + ctc (subword) } & 50.67 & 24.93 & 36.06 & 37.70 & 41.77 & 52.53 \\
    + LM & \textbf{44.67} & \textbf{21.31} & \textbf{33.57} & \textbf{36.28} & \textbf{33.83} & \textbf{45.50} \\
    \midrule
    \tabincell{l}{wav2vec2 encoder big \\ \quad + ctc (letter)} & 42.44 & 17.65 & 28.75 & 28.69 & 40.27 & 47.36 \\
    + LM & \textbf{35.62} & \textbf{16.07} & \textbf{28.16} & \textbf{28.32} & \textbf{25.70} & \textbf{39.11} \\
    relative improvement & 26.3\% & \textbf{52.4
    }\% & 25.1\% & 23.4\% & 42.9\% & 24.1\% \\
    \bottomrule
\end{tabular}
