# Стратегия `observed_data_likelihood`
**Назначение:** описать, какую задачу решает стратегия `OBSERVED_DATA_LIKELIHOOD`, показать, почему её блок зависит от всех компонент смеси, и объяснить, как задача вычисляется в коде.

## Задача

Блок стратегии `OBSERVED_DATA_LIKELIHOOD` для компоненты $q$ максимизирует не $Q$, а само {term}`правдоподобие наблюдаемых данных <Правдоподобие наблюдаемых данных>` {eq}`eq-observed-loglik` по параметрам блока. Веса и параметры остальных компонент фиксированы. Обозначим вклад остальных компонент в плотность смеси в точке $x_i$:

$$
b_i = \sum_{j \ne q} \omega_j \, f_j(x_i \mid \theta_j).
$$

Тогда задача блока:

$$
\theta_q^{[s]} \leftarrow \arg\max_{\theta_q^{[s]}} \ell(\theta_q), \qquad \ell(\theta_q) = \sum_{i=1}^{n} \ln\bigl(b_i + \omega_q \, f_q(x_i \mid \theta_q)\bigr).
$$ (eq-odl-target)

Ответственности $\gamma^{(t)}$ в {eq}`eq-odl-target` не входят: стратегия не использует `state.H`. Такие шаги составляют ECME {footcite:p}`liu1994ecme`: часть CM-шагов в нём максимизирует не $Q$, а правдоподобие при тех же ограничениях.

(math-odl-gradient)=
## Зависимость от других компонент

От $\theta_q$ в {eq}`eq-odl-target` зависит только слагаемое $\omega_q f_q$, поэтому

$$
\nabla_{\theta_q} \ell(\theta_q) = \sum_{i=1}^{n} \frac{\omega_q \, \nabla_{\theta_q} f_q(x_i \mid \theta_q)}{b_i + \omega_q \, f_q(x_i \mid \theta_q)}.
$$

Подставим

$$
\nabla_{\theta_q} f_q = f_q \, \nabla_{\theta_q} \ln f_q.
$$

Множитель перед $\nabla \ln f_q$ — ответственность {eq}`eq-responsibilities`, но вычисленная при текущем, а не зафиксированном на E-шаге $\theta_q$:

$$
\nabla_{\theta_q} \ell(\theta_q) = \sum_{i=1}^{n} \gamma_{iq}(\theta_q) \, \nabla_{\theta_q} \ln f_q(x_i \mid \theta_q), \qquad \gamma_{iq}(\theta_q) = \frac{\omega_q \, f_q(x_i \mid \theta_q)}{b_i + \omega_q \, f_q(x_i \mid \theta_q)}.
$$ (eq-odl-gradient)

Для сравнения градиент целевой функции стратегии [`q_function`](q_function.md) {eq}`eq-qj`:

$$
\nabla_{\theta_q} Q_q(\theta_q) = \sum_{i=1}^{n} \gamma_{iq}^{(t)} \, \nabla_{\theta_q} \ln f_q(x_i \mid \theta_q).
$$

Слагаемые те же, различаются веса. В $\nabla Q_q$ веса $\gamma_{iq}^{(t)}$ зафиксированы на E-шаге. В {eq}`eq-odl-gradient` веса $\gamma_{iq}(\theta_q)$ меняются вместе с $\theta_q$ и через $b_i$ зависят от параметров и весов всех остальных компонент. Поэтому:

- условие максимума {eq}`eq-odl-target` по $\theta_q$ зависит от значений остальных компонент в момент вычисления блока;
- блок `OBSERVED_DATA_LIKELIHOOD` читает все компоненты, в `STRATEGY_ACCESS` он `ALL_COMPONENTS` (см. {ref}`math-strategy-access`);
- блоки этой стратегии для разных компонент нельзя вычислять из одного снимка: результат зависит от порядка, и `MaximizationStep` выполняет их по одному в порядке списка.

## Вычисление в коде

| Символ | В коде |
|---|---|
| $\ln \omega_j + \ln f_j(x_i \mid \theta_j)$ | `mixture.component_lpdf_matrix(X) + log_weights` |
| $\ln b_i$ | `logsumexp` по строке той же матрицы, в которой столбец $q$ заменён на $-\infty$ |
| $\ln\bigl(b_i + \omega_q f_q(x_i \mid \theta_q)\bigr)$ | `np.logaddexp(log_background, log_target_weight + lpdf(candidate, X))` |
| $-\ell(\theta_q)$ | целевая функция, которую минимизирует `optimizer` |

Значения $\ln b_i$ вычисляются один раз до оптимизации: внутри блока параметры других компонент не меняются. Как и у численного пути `q_function`, недопустимые параметры дают $+\infty$, а поиск начинается с текущих значений.

От {eq}`eq-odl-target` целевая функция отличается одним: логарифм плотности смеси в каждой точке ограничен снизу константой `LOG_FLOOR`. Фактически максимизируется

$$
\tilde{\ell}(\theta_q) = \sum_{i=1}^{n} \max\Bigl(\ln\bigl(b_i + \omega_q \, f_q(x_i \mid \theta_q)\bigr), \; -10^{4}\Bigr).
$$ (eq-odl-floor)

Без ограничения одна точка вне носителя кандидата давала бы слагаемое $\ln 0 = -\infty$, и вся целевая функция становилась бы бесконечной. С ограничением такая точка даёт конечный штраф, и оптимизатор может сравнивать кандидатов.

## Монотонность шага

Если оптимизатор не возвращает точку хуже стартовой, блок не уменьшает {eq}`eq-odl-floor`. Пока ограничение не срабатывает ни в одной точке, $\tilde{\ell} = \ell$, и шаг не уменьшает правдоподобие наблюдаемых данных. Когда ограничение срабатывает, рост $\tilde{\ell}$ не гарантирует роста $\ln L$.

Отдельный шаг не уменьшает $\ln L$, но это ещё не доказывает монотонность всей итерации ECME, в которой такие шаги чередуются с шагами по $Q(\cdot \mid \Psi^{(t)})$. Этот вопрос относится к алгоритму ECME, а не к стратегии.

## Литература

```{footbibliography}
```
