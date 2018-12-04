from sharp.config.filters import Butterworth, Cheby1, Cheby2, WindowedSincFIR

# kwargs for SearchLines_BPF
# --------------------------

main_comp = dict(
    filename="main-comp",
    filters={
        "Butterworth": Butterworth(),
        "Chebyshev Type I": Cheby1(),
        "Chebyshev Type II": Cheby2(),
        "Windowed Sinc FIR": WindowedSincFIR(),
    },
)

cheby1_comp = dict(
    filename="cheby1-comp",
    filters={
        "0.1 dB": Cheby1(max_passband_atten=0.1),
        "1 dB": Cheby1(max_passband_atten=1),
        "2 dB": Cheby1(max_passband_atten=2),
        "4 dB": Cheby1(max_passband_atten=4),
        "8 dB": Cheby1(max_passband_atten=8),
        "16 dB": Cheby1(max_passband_atten=16),
    },
)

cheby2_comp = dict(
    filename="cheby2-comp",
    filters={
        "4 dB": Cheby2(min_stopband_atten=4),
        "10 dB": Cheby2(min_stopband_atten=10),
        "20 dB": Cheby2(min_stopband_atten=20),
        "40 dB": Cheby2(min_stopband_atten=40),
        "80 dB": Cheby2(min_stopband_atten=80),
        "120 dB": Cheby2(min_stopband_atten=120),
    },
)

sinc_FIR_comp = dict(
    filename="sinc-FIR-comp",
    filters={
        "1 %": WindowedSincFIR(transition_width=0.01),
        "2 %": WindowedSincFIR(transition_width=0.02),
        "5 %": WindowedSincFIR(transition_width=0.05),
        "10 %": WindowedSincFIR(transition_width=0.10),
        "20 %": WindowedSincFIR(transition_width=0.20),
        "40 %": WindowedSincFIR(transition_width=0.40),
    },
)
