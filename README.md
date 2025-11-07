# conv_fit_libreria.py

Libreria Python per l’analisi di dati di fotoluminescenza risolta nel tempo (TRPL) mediante convoluzione tra IRF sperimentale e decadimenti esponenziali (mono- e bi-esponenziali).

La libreria contiene funzioni per:

- caricare da file CSV i dati di PL e l'IRF;
- normalizzare l'IRF;
- costruire modelli teorici basati su convoluzione `IRF *` decadimento esponenziale (mono e bi-esponenziale);
- eseguire fit non lineari (mono e bi-esponenziale) con stima delle incertezze;
- gestire i risultati tramite dataclass (`MonoExpFitResult`, `BiExpFitResult`);
- generare grafici dei dati e della curva di fit.

## Import tipico

Nel codice Python si può importare la libreria ad esempio con:

```python
from conv_fit_libreria import (
    load_time_resolved_csv,
    normalize_irf,
    estimate_sampling_interval,
    monoexp_convolution_model,
    biexponential_convolution_model,
    fit_monoexponential_convolution,
    fit_biexponential_convolution,
    plot_fit,
    MonoExpFitResult,
    BiExpFitResult,
)
```

---

## Strutture dati dei risultati

I risultati dei fit vengono restituiti come dataclass (decoratore `@dataclass`).

### `MonoExpFitResult`

Campi principali:

- `tau` (float): tempo di vita stimato (nelle stesse unità dell'asse temporale, ad esempio ns).
- `amplitude` (float): coefficiente di scala ottimale A del modello convoluto.
- `offset` (float): termine di fondo B.
- `time` (`np.ndarray`): asse dei tempi dei dati misurati.
- `model` (`np.ndarray`): modello convoluto IRF * esponenziale (prima di applicare A e B).
- `fitted` (`np.ndarray`): curva di fit `A * model + B`.
- `residuals` (`np.ndarray`): residui `y_meas - fitted`.
- `optimizer` (`OptimizeResult`): oggetto restituito da `scipy.optimize.least_squares`.
- `statistics` (`Dict[str, Any]`): dizionario con informazioni statistiche (gradi di libertà, varianza stimata, errori standard e intervalli di confidenza per tau, A e B, matrici di covarianza e correlazione).
- `sse` (float): somma dei quadrati dei residui (sum of squared errors).
- `sst` (float): somma dei quadrati totali rispetto alla media dei dati.
- `r_squared` (float): coefficiente di determinazione R².

### `BiExpFitResult`

Campi principali:

- `tau1`, `tau2` (float): tempi di vita delle due componenti esponenziali.
- `alpha` (float): peso della prima componente. Il kernel interno è del tipo  
  `h(t) = alpha * h1(t; tau1) + (1 - alpha) * h2(t; tau2)`.
- `amplitude`, `offset`, `time`, `model`, `fitted`, `residuals`, `optimizer`,
  `statistics`, `sse`, `sst`, `r_squared`: analoghi al caso mono-esponenziale.

---

## Caricamento dati e normalizzazione IRF

### `load_time_resolved_csv`

```python
t_pl, pl_signal, t_irf, irf_signal = load_time_resolved_csv(
    "dati_trpl.csv",
    delimiter=",",
    skip_header=1,
)
```

La funzione si aspetta un file CSV con le seguenti colonne:

- colonna 0: asse dei tempi PL;
- colonna 1: intensità PL;
- colonna 4: asse dei tempi IRF;
- colonna 5: intensità IRF.

Restituisce quattro array NumPy: `t_pl`, `pl_signal`, `t_irf`, `irf_signal`.

### `normalize_irf`

```python
t_irf_norm, irf_norm, irf_area = normalize_irf(t_irf, irf_signal)
```

Questa funzione:

- lavora su `t_irf` e `irf_signal`;
- opzionalmente può inserire un punto extra `(time, counts)` tramite l'argomento `add_point`;
- calcola l'area dell'IRF con il metodo di Simpson;
- normalizza i conteggi in modo che il massimo valga 1.

Restituisce:

- `t_irf_norm`: asse dei tempi (eventualmente esteso con il punto aggiunto);
- `irf_norm`: IRF normalizzata (massimo pari a 1);
- `irf_area`: area dell'IRF originale.

### `estimate_sampling_interval`

```python
dt = estimate_sampling_interval(t_pl)
```

Stima il passo di campionamento Δt come mediana delle differenze successive dei tempi.
Viene usata quando il parametro `dt` non viene fornito esplicitamente alle funzioni di modello e di fit.

---

## Modelli di convoluzione

### Kernel esponenziale causale

Il kernel esponenziale causale continuo ha la forma:

- `h(t) = (1/tau) * exp(-(t - t_shift) / tau)` per `t >= t_shift`
- `h(t) = 0` per `t < t_shift`

Il parametro `shift` (qui indicato con `t_shift`) rappresenta il ritardo tra il picco dell'IRF e l'inizio del decadimento.

Internamente, il kernel discreto viene costruito su una griglia temporale relativa (`t_rel`) tramite la funzione interna `_causal_exponential_kernel`.

### `monoexp_convolution_model`

```python
model = monoexp_convolution_model(
    t_irf_norm,
    irf_norm,
    t_eval=t_pl,
    tau=1.0,   # valore o guess di tau
    dt=None,   # se None, viene stimato da t_irf_norm
    shift=0.0, # tipicamente 0
)
```

La funzione:

1. costruisce il kernel esponenziale causale sui tempi relativi dell'IRF;
2. esegue la convoluzione `IRF * kernel` tramite `scipy.signal.fftconvolve`;
3. interpola il risultato sulla griglia temporale `t_eval` (di solito l'asse dei tempi dei dati PL).

Restituisce un array `model` della stessa lunghezza di `t_eval`.

### `biexponential_convolution_model`

```python
model_bi = biexponential_convolution_model(
    t_irf_norm,
    irf_norm,
    t_eval=t_pl,
    tau1=0.5,
    tau2=3.0,
    alpha=0.2,
    dt=None,
    shift=0.0,
)
```

In questo caso il kernel è una combinazione di due esponenziali causali:

```text
h(t) = alpha * h1(t; tau1, t_shift) + (1 - alpha) * h2(t; tau2, t_shift)
```

Anche qui viene usata `fftconvolve` e il risultato viene interpolato su `t_eval`.

---

## Fit mono-esponenziale

### `fit_monoexponential_convolution`

Questa è la funzione principale per eseguire il fit mono-esponenziale convoluto.

```python
result = fit_monoexponential_convolution(
    t_meas=t_pl_window,      # finestra temporale da fittare
    y_meas=pl_window,        # intensità PL sulla stessa finestra
    t_irf=t_irf_norm,        # asse dei tempi IRF
    irf_signal=irf_norm,     # IRF normalizzata (max=1)
    tau_guess=1.0,           # guess iniziale per tau
    tau_bounds=(1e-3, 20.0), # limiti fisici per tau
    shift=0.0,               # ritardo fissato
    dt=None,                 # se None, stimato da t_irf_norm
    max_nfev=500,            # massimo numero di valutazioni
    sigma=None,              # pesi opzionali per WLS
    p_total=3,               # tau, A, B
    alpha=0.05,              # livello di significatività (CI 95%)
)
```

Il modello che viene fittato ai dati è:

```text
y_fit(t) = A * (IRF * h_tau)(t) + B
```

Il parametro non lineare è `tau`, mentre `A` e `B` vengono determinati ad ogni passo in forma chiusa tramite `best_affine_scaling`.

#### Accesso ai risultati del fit mono-esponenziale

```python
print("tau =", result.tau, "ns")
print("A   =", result.amplitude)
print("B   =", result.offset)
print("R^2 =", result.r_squared)

tau_info = result.statistics["tau"]
print("tau =", tau_info["value"], "+/-", tau_info["se"], "ns")
print("95% CI:", tau_info["ci"])
```

#### Esempio completo: fit mono-esponenziale su una finestra

```python
from conv_fit_libreria import (
    load_time_resolved_csv,
    normalize_irf,
    fit_monoexponential_convolution,
    plot_fit,
)
import matplotlib.pyplot as plt

# 1) Carica dati
t_pl, pl_signal, t_irf, irf_signal = load_time_resolved_csv("dati_trpl.csv")

# 2) Normalizza IRF
t_irf_norm, irf_norm, irf_area = normalize_irf(t_irf, irf_signal)

# 3) Seleziona una finestra temporale
i_start = 5000
i_end   = 15000
t_window  = t_pl[i_start:i_end]
pl_window = pl_signal[i_start:i_end]

# 4) Fit mono-esponenziale convoluto
res_mono = fit_monoexponential_convolution(
    t_meas=t_window,
    y_meas=pl_window,
    t_irf=t_irf_norm,
    irf_signal=irf_norm,
    tau_guess=1.0,
    tau_bounds=(1e-3, 20.0),
    shift=0.0,
)

print("tau (mono) =", res_mono.tau, "ns")
print("R^2        =", res_mono.r_squared)

# 5) Plot del risultato
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_fit(
    ax,
    res_mono.time,
    pl_window,
    res_mono.fitted,
    data_label="Dati",
    fit_label="Fit monoexp",
)
ax.set_xlabel("t [ns]")
ax.set_ylabel("Intensità PL (arb. units)")
plt.tight_layout()
plt.show()
```

---

## Fit bi-esponenziale

### `fit_biexponential_convolution`

Per modellare decadimenti con due componenti:

```python
res_bi = fit_biexponential_convolution(
    t_meas=t_window,
    y_meas=pl_window,
    t_irf=t_irf_norm,
    irf_signal=irf_norm,
    initial=(0.5, 3.0, 0.2),  # (tau1, tau2, alpha) guess iniziali
    bounds=(
        (1e-3, 1e-3, 0.0),     # lower bounds
        (20.0, 20.0, 1.0),     # upper bounds
    ),
    shift=0.0,
    dt=None,
    max_nfev=5000,
    sigma=None,
    p_total=5,                # tau1, tau2, alpha, A, B
    alpha=0.05,
    names=("tau1", "tau2", "alpha"),
)
```

#### Accesso ai risultati del fit bi-esponenziale

```python
print("tau1  =", res_bi.tau1, "ns")
print("tau2  =", res_bi.tau2, "ns")
print("alpha =", res_bi.alpha)
print("R^2   =", res_bi.r_squared)

tau1_info = res_bi.statistics["tau1"]
tau2_info = res_bi.statistics["tau2"]
alpha_info = res_bi.statistics["alpha"]

print("tau1  =", tau1_info["value"],  "+/-", tau1_info["se"])
print("tau2  =", tau2_info["value"],  "+/-", tau2_info["se"])
print("alpha =", alpha_info["value"], "+/-", alpha_info["se"])
```

#### Plot del fit bi-esponenziale

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_fit(
    ax,
    res_bi.time,
    pl_window,
    res_bi.fitted,
    data_label="Dati",
    fit_label="Fit biexp",
)
ax.set_xlabel("t [ns]")
ax.set_ylabel("Intensità PL (arb. units)")
plt.tight_layout()
plt.show()
```

---

## Note pratiche

- **Parametro `shift`**  
  Di solito, se IRF e PL sono già allineate, si usa `shift = 0.0` e lo si tiene fissato.  
  Per un sistema causale si osserva che:
  - per `tshift < 0` il fit non cambia: l’effetto viene assorbito nel parametro di scala `A`;
  - per `tshift > 0` compaiono artefatti, in particolare una traslazione “in avanti” della curva fittata.  

  Questo perché, fisicamente, `tshift` rappresenta il ritardo tra il picco dell’IRF e l’inizio della risposta (ad esempio il decadimento). Un `tshift` positivo implica che il sistema inizi a decadere dopo il picco dell’IRF, cosa che non avviene nel caso considerato, e che il fit non riesce a compensare tramite `A` come invece avviene per `tshift` negativo. Dal punto di vista matematico, il comportamento è legato alla presenza della funzione di Heaviside nel kernel, che dipende da `(t - t_shift)` e dal fatto che l’asse dei tempi parte già da zero.

- **Prestazioni**  
  La convoluzione viene effettuata con `scipy.signal.fftconvolve`, che è efficiente anche per segnali lunghi.  
  Il tempo di fit cresce con:
  - il numero di punti temporali,
  - il numero di valutazioni di funzione (`nfev`).

- **Pesi `sigma`**  
  Se sono note le incertezze punto per punto dei dati PL, si possono passare tramite il parametro `sigma`, ottenendo un fit ai minimi quadrati pesati (WLS).
