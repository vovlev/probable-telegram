import pandas as pd
import bisect
import seaborn as sns
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from IPython.display import display, clear_output, Image
import scipy
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import scipy.stats
from scipy.special import logit
from scipy.stats import chi2
import plotly.graph_objects as go
import plotly.express as px
import warnings
import matplotlib.colors as mcolors

from functools import wraps


def _time_decorator(time_info, mode):
    """
    Декоратор для автоматического приведения даты к pd.datetime.
    Работает в двух режимах:

    "column"  : когда аргумент со временем – это pd.Series (e.g. woe_stab)
        Тогда в  time_info достаточно подать словарь

        d = {
            "index" : x1,  # индекс неименованного (без дефолтного значения) аргумента, int
            "name" : x2,  # название аргумента, str – для всех
        }

    "column_name" : когда аргумент со временем – это название столбца (e.g. roc_auc_time)
        Тогда добавляем time_info "df_index" и "df_name" – информацию об аргументе для датафрейма
        по аналогии с "index", "name".
    """

    def inside_decorator(func):

        @wraps(func)  # декоратор сохраняет
        def F(*args, transform_dt=False, **kwargs):

            if transform_dt:

                index, name = time_info["index"], time_info["name"]

                # если аргумент со временем – столбец
                if mode == "column":

                    # аргумент со временем среди дефолтных
                    if len(args) - 1 >= index:
                        args = list(args)  # чтобы можно было делать item assignment
                        args[index] = pd.to_datetime(args[index])

                    # аргумент со временем среди именованных
                    elif name in kwargs.keys():
                        kwargs[name] = pd.to_datetime(kwargs[name])

                    # если аргумент со временем не нашелся
                    # то ничего не делаем, т.к. функция должна и так лечь :)
                    else:
                        pass

                # если аргумент со временем – название столбца в df
                elif mode == "column_nm":
                    df_index, df_name = time_info["df_index"], time_info["df_name"]
                    try:
                        column_nm = args[index] if (len(args) - 1 >= index) else kwargs[name]

                        # аргумент со временем среди дефолтных
                        if len(args) - 1 >= df_index:
                            args[df_index][column_nm] = pd.to_datetime(args[df_index][column_nm])

                        # аргумент со временем среди именованных
                        elif df_name in kwargs.keys():
                            kwargs[df_name][column_nm] = pd.to_datetime(kwargs[df_name][column_nm])

                        # если аргумент со временем не нашелся
                        # то ничего не делаем, т.к. функция должна и так лечь :)
                        else:
                            pass

                    # если столбец name не в ключах kwargs
                    except KeyError:
                        pass

                else:
                    pass

            return func(*args, **kwargs)

        return F

    return inside_decorator


def _format_val(x, precision=3):
    """format a value for _make_buck
    >>> _format_val(0.00001)
    '1e-05'
    >>> _format_val(2.00001)
    '2.0'
    >>> _format_val(1000.0)
    '1000'
    >>> _format_val('foo')
    'foo'
    """
    if isinstance(x, float):
        if np.equal(np.mod(x, 1), 0):
            return '%d' % x
        if not np.isfinite(x):
            return '%s' % x
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return '%s' % np.around(x, digits)
    return '%s' % x


def _length_of_buckets(arr_sorted, n_buck, verbose=False):
    """
    Возвращает длины бакетов по порядку. Входящий массив должен быть отсортирован по возрастанию.
    Функция работает рекурсивно, на каждой итерации находя длину самого левого бакета из еще неисследованной части массива
    Примеры:
    _length_of_buckets(np.array(range(1, 13)), 3) == [4, 4, 4]
    _length_of_buckets(np.array([1, 2, 3]), 3) == [1, 1, 1]
    _length_of_buckets(np.array([1, 2, 3, 4, 4, 4, 4, 4]), 2) == [3, 5]
    _length_of_buckets(np.array([1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6]), 3) == [3, 9, 3]
    """
    n = len(arr_sorted)

    if n_buck == 1:
        return [n]
    elif n_buck == 0:
        return []
    elif n_buck > n:
        print(arr_sorted, n_buck, n)
        raise ValueError('Число уникальных значений в массиве меньше числа бакетов')

    if verbose:
        print('n: ', n)
        print('n_buck: ', n_buck)
        print('n // n_buck: ', n // n_buck)

    def bisect_bounds(len_bucket_tmp):
        bound_value = arr_sorted[len_bucket_tmp]
        left, right = bisect.bisect_left(arr_sorted, bound_value), bisect.bisect_right(arr_sorted, bound_value)
        return left, right

    def set_bounds(left_tmp, right_tmp):
        l, r = bisect_bounds(left_tmp)
        if l == 0:
            left_len_bucket = r
        else:
            left_len_bucket = l
        l, r = bisect_bounds(right_tmp)
        if r == n:
            right_len_bucket = n - l
        else:
            right_len_bucket = n - r
        # up - вынести Counter из рекурсии
        center_unique = len(Counter(arr_sorted[left_len_bucket:n - right_len_bucket]))
        if center_unique < n_buck - 2:
            # up - вынести Counter из рекурсии
            if len(Counter(arr_sorted[:left_len_bucket])) <= 1 and len(Counter(arr_sorted[n - right_len_bucket:])) <= 1:
                if verbose:
                    print(arr_sorted, n_buck, n)
                raise ValueError('Число уникальных значений в массиве меньше числа бакетов')
            return set_bounds(left_len_bucket - 1, n - right_len_bucket + 1)
        return left_len_bucket, right_len_bucket

    left_len_bucket, right_len_bucket = set_bounds(n // n_buck, n - n // n_buck)

    if n_buck == 2:
        right_len_bucket = n - left_len_bucket

    if verbose:
        print('left_len_bucket: ', left_len_bucket)
        print('right_len_bucket: ', right_len_bucket)
    return [left_len_bucket] + _length_of_buckets(arr_sorted[left_len_bucket:n - right_len_bucket],
                                                  n_buck - 2,
                                                  verbose=verbose) + [right_len_bucket]


def _make_bucket(series, num_buck, method='quantile', verbose=False):
    """
    Функция для разбивки на бакеты. Бьём непустые значения на num_buck бакетов
    """

    unique = pd.Series(series.unique())
    if len(unique.dropna()) < num_buck:
        raise ValueError('Количество уникальных непустых значений в массиве меньше числа бакетов')

    nan_cnt = series.isna().sum()

    if method == 'dense':
        bucket = np.ceil(series.rank(pct=True, method='dense') * num_buck).fillna(num_buck + 1)
        bucket = pd.Categorical(bucket, categories=np.sort(bucket.unique()), ordered=True)

    elif method == 'quantile':
        ser_no_na = series.dropna()
        if verbose:
            print('ser_no_na: ', ser_no_na)
        ser_sorted = ser_no_na.sort_values()
        if verbose:
            display('ser_sorted: ', ser_sorted)
        length_of_buckets = _length_of_buckets(ser_sorted.array, num_buck, verbose=verbose)
        if verbose:
            display('length_of_buckets: ', length_of_buckets)
        pairs = zip(range(1, num_buck + 1), length_of_buckets)
        if verbose:
            display('pairs: ', pairs)

        bucket_values = [[x] * y for x, y in zip(range(1, num_buck + 1), length_of_buckets)]
        if verbose:
            display('len(bucket_values): ', len(bucket_values))

        ser_replaced = pd.Series(np.concatenate(bucket_values), index=ser_sorted.index)
        if verbose:
            display('ser_replaced: ', ser_replaced)
        bucket = pd.concat([series, ser_replaced], axis=1).iloc[:, 1]
        if verbose:
            display('bucket: ', bucket)
        bucket = bucket.fillna(num_buck + 1)
        bucket = pd.Categorical(bucket, categories=np.sort(bucket.unique()), ordered=True)
        if verbose:
            display('bucket: ', bucket)
    else:
        raise ValueError(f'Wrong method "{method}", use quantile or dense')
    agg = series.groupby(bucket).agg(['min', 'max'])
    if verbose:
        display('agg: ', agg)

    def _format_buck(row, precision=3):
        if row.name == num_buck + 1:
            return 'missing'
        if row['min'] == row['max']:
            return _format_val(row['min'], precision)
        return '[{}; {}]'.format(
            _format_val(row['min'], precision),
            _format_val(row['max'], precision)
        )

    # при округлении до трех знаков иногда получаются не уникальные значения границ бакетов.
    # для таких кейсов увеличиваем точность, пока не получатся уникальные границы
    # 17 выбрано в соответствии с максимальной точностью числа float
    for p in range(3, 17):
        try:
            names = agg.apply(_format_buck, args=(p,), axis=1)
            if verbose:
                display('names: ', names)
            res = bucket.rename_categories(names.to_dict())
            if verbose:
                display('res: ', res)
            break
        except ValueError:
            if verbose:
                print(f'precision {p} is not enough')

    return res


def _clopper_pearson(k, n, alpha=0.32):
    # Функция для поиска доверительных интервалов
    lo = scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    lo = np.nan_to_num(lo)
    hi = 1 - np.nan_to_num(1 - hi)
    return lo, hi


def _get_predicted_values_linreg(
        target: np.ndarray,
        feature: np.ndarray
):
    '''Функция возвращает предсказанные значения для обучающей выборки'''
    reg = LinearRegression()
    reg.fit(feature.reshape(-1, 1), target)
    predicted_values = reg.predict(feature.reshape(-1, 1))
    return predicted_values


def _get_r2_score_lingreg(
        target: np.ndarray,
        feature: np.ndarray
):
    '''Функция возвращает R_sqr для линейной регрессии'''
    predicted_values = _get_predicted_values_linreg(target, feature)
    r_score = r2_score(target, predicted_values)
    return r_score


def _get_logreg(
        df_tmp: pd.DataFrame,
        df_stat: pd.DataFrame
):
    '''Функция возвращает значения линии logit по бакетам и первый коэффициент в logit'''
    scaler = StandardScaler()
    scaler.fit(df_tmp[['var']])
    clf = LogisticRegression(penalty='none', solver='lbfgs', max_iter=500)
    clf.fit(scaler.transform(df_tmp[['var']]), df_tmp['target'])

    # Вероятности принадлежности к классу для бакетов
    predicted_prob = clf.predict_proba(scaler.transform(df_stat[['var']]))[:, 1]
    # Массив из значения вероятности дефолта по всей выборке (количество бакетов задает размер массива)
    total_ratio = np.repeat(df_tmp['target'].mean(), df_stat.shape[0])
    total_ratio = np.clip(total_ratio, 0.001, 0.999)
    # Значения логита по бакетам
    logreg = logit(predicted_prob) - logit(total_ratio) # заменил
    return logreg, clf.coef_[0][0]


def _get_res_logreg(
        df_tmp: pd.DataFrame,
        df_stat: pd.DataFrame
):
    '''Функция возвращает ROC_AUC, IV, R_sqr, n_obs, n_buckets, coef[0] для логистической регрессии'''
    logreg, coef = _get_logreg(df_tmp, df_stat)
    target_count = df_stat['target_count'].values
    obj_count = df_stat['obj_count'].values
    woe = df_stat['woe'].values

    n_obs = df_tmp.shape[0]
    bad = df_tmp['target'].sum()
    good = n_obs - bad
    n_buckets = df_stat.bucket.nunique()

    IV = np.sum(((target_count / bad - (obj_count - target_count) / good) * (df_stat['woe']))) # заменил
    R_sqr = 1 - np.sum(obj_count * (woe - logreg) ** 2) / (
            np.sum(obj_count * (woe) ** 2) - np.sum(
        obj_count * woe) ** 2 / np.sum(obj_count)) # заменил v02
    auc_raw = roc_auc_score(df_tmp['target'], df_tmp['var'])
    auc = max(auc_raw, 1 - auc_raw)
    return {
        'AUC': round(auc, 3),
        'IV': round(IV, 3),
        'R_sqr': round(R_sqr, 3),
        'n_obs': n_obs,
        'n_buckets': n_buckets,
        'coef': coef
    }


def _make_confidence_interval(
        *,
        df_stat,
        is_categorical_target,
        interval_type,
        confidence,
        n_obs,
        bad,
        warnings_method
):
    """
        Расчет woe и границ woe для бинарного таргета
        Расчет квантилей и доверительных интервалов для непрерывного таргета

        n_obs и bad нужны для бинарного таргета
    """
    if is_categorical_target:
        values_checked = np.array([_check_bads_count(obj_count, target_count, True, bucket, method=warnings_method)
                                   for obj_count, target_count, bucket in
                                   zip(df_stat['obj_count'], df_stat['target_count'], df_stat['bucket'])]).T
        df_stat['obj_count'] = values_checked[0]
        df_stat['target_count'] = values_checked[1]

        df_stat = df_stat.assign(obj_rate=lambda x: x['obj_count'] / n_obs,
                                 target_rate=lambda x: x['target_count'] / x['obj_count'],
                                 target_rate_total=lambda x: bad / n_obs)
        df_stat = df_stat.assign(woe=lambda x: round(logit(x['target_rate']) - logit(x['target_rate_total']), 3), # заменил
                                 woe_max=lambda x: round(
                                     logit(_clopper_pearson(x['target_count'], x['obj_count'])[0]) - logit(
                                         x['target_rate_total']), 3), # заменил
                                 woe_min=lambda x: round(
                                     logit(_clopper_pearson(x['target_count'], x['obj_count'])[1]) - logit(
                                         x['target_rate_total']), 3), # заменил
                                 )
        df_stat = df_stat.assign(woe_l=lambda x: x['woe'] - x['woe_min'],
                                 woe_h=lambda x: x['woe_max'] - x['woe'],
                                 )
    else:
        if interval_type == 'mean':
            df_stat = df_stat.assign(target_interval=lambda x: x['sem'] * scipy.stats.t.ppf(
                (1 + confidence) / 2., x['obj_count'] - 1))
        elif interval_type == 'quantile':
            df_stat = df_stat.assign(quantile_l=lambda x: x['target_mean'] - x['qntl_25'],
                                     quantile_h=lambda x: x['qntl_75'] - x['target_mean'])

        else:
            return TypeError('Wrong interval type. Try mean or quantile')
    return df_stat


def _get_res_woe_line(
        *,
        df_tmp,
        df_stat,
        is_categorical_target,
        n_obs,
        n_buckets
):
    if is_categorical_target:
        res = _get_res_logreg(df_tmp, df_stat)
    else:
        res = {
            'R_sqr': round(_get_r2_score_lingreg(df_tmp['target'].values,
                                                 df_tmp['var'].values), 3),
            'n_obs': n_obs,
            'n_buckets': n_buckets
        }
    return res


def _plot_line(
        *,
        df_tmp,
        df_stat,
        is_categorical_target,
        save_plot,
        legend_position,
        interactive_plot,
        plot_nm=None
):
    '''Функция для построения линии регрессии или интерполяции'''
    target = df_tmp['target'].values
    feature = df_tmp['var'].values
    stat_agg_feature = df_stat['var'].values

    interpolation_line_name = ''
    values_line_name = ''
    if is_categorical_target:
        predicted_target = _get_logreg(df_tmp, df_stat)[0]
        stat_agg_value = df_stat['woe'].values
        stat_feature = stat_agg_feature
        border_h = df_stat['woe_h']
        border_l = df_stat['woe_l']
        interpolation_line_name = 'Interpolation'
        values_line_name = 'WoE'

    else:
        predicted_target = _get_predicted_values_linreg(target, feature)
        stat_agg_value = df_stat['target_mean'].values
        stat_feature = feature
        border_h = df_stat['quantile_h']
        border_l = df_stat['quantile_l']
        interpolation_line_name = 'Linear Regression'
        values_line_name = 'monthly_income_amt'

    fig = go.Figure()

    # общие настройки
    title = dict(
        text=plot_nm,
        y=0.95,
        x=0.5,
        font=dict(size=12),
        xanchor="center",
        yanchor="top"
    )
    margin = go.layout.Margin(
        l=50,  # left margin
        r=50,  # right margin
        b=50,  # bottom margin
        t=60  # top margin
    )

    fig.add_trace(
        go.Scatter(
            x=stat_feature,
            y=predicted_target,
            mode='lines',
            name=interpolation_line_name
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stat_agg_feature,
            y=stat_agg_value,
            line=dict(
                color='firebrick',
                width=1,
                dash='dot'
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=border_h,
                arrayminus=border_l
            ),
            name=values_line_name)
    )

    if save_plot:
        width, height = 690, 470

    else:
        width, height = 1000, 450

    # если legend_position задано явно
    if legend_position:
        legend_position_y, legend_position_x = legend_position.split()

    # иначе: "top left" если auc_raw (который до взяти max(auc_raw, 1 - auc_raw) >= 0.5 иначе "top right"
    else:
        legend_position_y = "top"
        legend_position_x = "right"

    legend_y = 0.01 if legend_position_y == 'bottom' else 0.99 if legend_position_y == 'top' else None
    legend_x = 0.01 if legend_position_x == 'left' else 0.99 if legend_position_x == 'right' else None

    fig.update_layout(
        width=width,  # 760,
        height=height,  # 535,
        xaxis_title=dict(
            text='Feature value',
            font=dict(size=12)
        ),
        yaxis_title=dict(
            text=values_line_name,
            font=dict(size=12)
        ),
        legend=dict(
            x=legend_x,
            y=legend_y,
            font=dict(size=12),
            xanchor=legend_position_x,
            yanchor=legend_position_y
        ),
        title=title,
        margin=margin
    )
    if interactive_plot:
        fig.show()
    else:
        img_bytes = fig.to_image(format="png")
        display(Image(img_bytes))
    return fig


def _plot_hist(
        *,
        df_tmp,
        hist_scale='count',
        plot_nm='plot name',
        interactive_plot=True,
        is_categorical_target=True,
        save_plot=False
):
    '''
        Функция для построения гистограммы
    '''

    margin = go.layout.Margin(
        l=50,  # left margin
        r=50,  # right margin
        b=50,  # bottom margin
        t=60  # top margin
    )

    title = dict(
        text=plot_nm,
        y=0.95,
        x=0.5,
        font=dict(size=12),
        xanchor="center",
        yanchor="top"
    )

    if hist_scale == 'count':
        hist_color = 'bucket'
        hist_norm = None
    elif hist_scale == 'percent':
        hist_color = None
        hist_norm = 'percent'
    else:
        raise ValueError("hist_scale should be in {'count', 'percent'}")

    if save_plot:
        width, height = 690, 470
    else:
        width, height = 1000, 450

    if is_categorical_target:
        labels = None
    else:
        labels = {
            # 'var': feature_name,
            'bucket': 'Buckets'
        }

    fig_hist = px.histogram(
        df_tmp,
        x='var',
        color=hist_color,
        labels=labels,
        histnorm=hist_norm,
        width=width,
        height=height,
    )

    if is_categorical_target:
        fig_hist.update_layout(
            showlegend=False,
            title=title,
            margin=margin

        )
    if interactive_plot:
        fig_hist.show()
    else:
        img_bytes = fig_hist.to_image(format="png")
        display(Image(img_bytes))
    return fig_hist


def woe_line(
        var,
        target,
        n_buckets=None,
        var_nm=None,
        target_nm=None,
        plot_nm=None,
        is_category=False,
        is_categorical_target=True,
        interval_type='quantile',
        plot_line=True,
        legend_position=None,
        plot_hist=True,
        hist_scale='count',
        show_table=False,
        bins_method='quantile',
        verbose=False,
        confidence=0.99,
        return_plotly_object=False,
        show_plot=True,
        interactive_plot=False,
        save_plot=False,
        warnings_method='print',
        error_method='print'
):
    """
    Функция бьет переменную на бакеты и для каждого из них считает значение WoE
    с последующей отрисовкой на графике для проверки линейности

    Parameters
    ----------
    var : array-like
        массив с переменной
    target : array-like
        массив с целевой переменной
    n_buckets : int, default None
        кол-во бакетов, на которое нужно разбить переменную. Если не указано,
        то выбирается автоматически исходя из размера выборки
    var_nm : str, default None
        название признака
    target_nm : str, default None
        название целевой переменной
    plot_nm : str, default None
        название графика
    is_category : bool, default False
        флаг категориальной переменной
    is_categorical_target : bool, default True
        флаг бинарного таргета (is_categorical_target=False означает, что таргет непрерывный)
    interval_type: str, default 'quantile'
        параметр для непрерывного таргета
        тип интервалов. При значении 'mean' вычисляются доверительные интервалы для среднего,
        при значении 'quantile' - квантили 0.25 и 0.75
    plot_line : bool, default True
        флаг построения графика линейности
    legend_position : str, default None
        расположение легенды на первом графике - можно выбрать из "top" / "bottom" и "left" / "right"
        и задать одной строкой. Например, "top left" или "bottom right".

        Если не задано явно, `legend_position = "top left" if auc >= 0.5 else "top right"`
    plot_hist : bool, default True
        флаг построения гистограммы
    hist_scale : {'count', 'percent'}, default 'count'
        y-шкала гистограммы:

        - 'count':  в столбцах число наблюдений

        - 'percent' : в столбцах процент наблюдений от общего числа
    show_table : bool, default False
        флаг отображения таблицы с информацией о бакетах
    bins_method : str, default 'quantile'
        'quantile' или 'dense'
    verbose : str, default False
        флаг выведения подробных логов
    confidence : float, default 0.99
        уровень доверия для построения доверительных интервалов feature_line
    return_plotly_object : bool, default False
        флаг возврата графиков
    show_plot : bool, default True
        флаг отрисовки графиков
    interactive_plot : bool, default False
        флаг построения интерактивных графиков
    save_plot : bool, default False
        флаг сохранения графиков
    warnings_method: str, default 'print'
        уровень вывода предупреждений

            - 'print' : в обычный поток вывода
            - 'warning' : в поток вывода ошибок
            - 'silent' : не выводить предупреждения

    error_method: str, default 'print'
        уровень вывода ошибок

            - 'print' - вывод ошибок

    Returns
    ---
    res : dict
        словарь:
            ключи бинарного таргета:
                - 'AUC' : ROC-AUC предсказания
                - 'IV' : informational value
                - 'R_sqr' : коэффициент детерминации
                - 'n_obs' : количество наблюдений
                - 'n_buckets' : итоговое количество бакетов
                - 'coef' : коэффициент в логите

    """

    df_tmp = pd.DataFrame(data={'var': np.array(var), 'target': np.array(target)})
    df_tmp = _check_nan(df_tmp, method=warnings_method)

    n_obs = df_tmp['target'].count()
    bad = df_tmp['target'].sum()

    if n_buckets is None:
        n_buckets = min(
            len(df_tmp['var'].dropna().unique()),
            int(
                round(
                    np.power(len(df_tmp['var']), 0.33
                             ) / 3
                )
            )
        )

    # бьем на бакеты
    df_stat = None
    if not is_category:
        while n_buckets >= 1:
            try:
                df_tmp = df_tmp.assign(
                    bucket=lambda x: _make_bucket(x['var'], n_buckets, bins_method, verbose=verbose)
                )
                if is_categorical_target:
                    df_stat = df_tmp.groupby('bucket', as_index=False).agg(
                        target_count=('target', 'sum'),
                        obj_count=('target', 'size'),
                        var=('var', 'mean')
                    )
                else:
                    df_stat = df_tmp.groupby('bucket', as_index=False).agg(
                        target_mean=('target', 'mean'),
                        sem=('target', 'sem'),
                        obj_count=('target', 'size'),
                        qntl_25=('target', lambda x: np.quantile(x, 0.25)),
                        qntl_75=('target', lambda x: np.quantile(x, 0.75)),
                        var=('var', 'mean')
                    )
                break
            except (ValueError, IndexError):
                # если не получилось побить на указанное число бакетов, бьем на меньшее число
                if error_method == 'print':
                    print(f'n_buckets {n_buckets} reduced to {n_buckets - 1}')
                n_buckets -= 1

    else:
        df_stat = df_tmp.groupby('var', as_index=False).agg(
            target_count=('target', 'sum'),
            obj_count=('target', 'size')
        ).rename(columns={'var': 'bucket'})

    df_stat = _make_confidence_interval(
        df_stat=df_stat,
        is_categorical_target=is_categorical_target,
        interval_type=interval_type,
        confidence=confidence,
        n_obs=n_obs,
        bad=bad,
        warnings_method=warnings_method
    )
    if show_table:
        display(df_stat)

    if is_category:
        return df_stat
    else:
        res = _get_res_woe_line(
            df_tmp=df_tmp,
            df_stat=df_stat,
            is_categorical_target=is_categorical_target,
            n_obs=n_obs,
            n_buckets=n_buckets
        )

        fig, fig_hist = None, None
        if plot_line:
            if show_plot:
                if plot_nm is None:
                    if is_categorical_target:
                        plot_nm = f"{var_nm} | {target_nm} AUC = {res['AUC']}  IV = {res['IV']} R_sqr = {res['R_sqr']}"
                    else:
                        plot_nm = f" R_sqr = {res['R_sqr'] : .3f}"

            fig = _plot_line(
                df_tmp=df_tmp,
                df_stat=df_stat,
                is_categorical_target=is_categorical_target,
                save_plot=save_plot,
                legend_position=legend_position,
                interactive_plot=interactive_plot,
                plot_nm=plot_nm
            )
            if save_plot:
                fig.write_image("%s_%s_line.png" % (var_nm, target_nm))
        if plot_hist:
            if show_plot:
                fig_hist = _plot_hist(
                    df_tmp=df_tmp.sort_values('bucket'),
                    hist_scale=hist_scale,
                    plot_nm=plot_nm,
                    interactive_plot=interactive_plot,
                    is_categorical_target=is_categorical_target,
                    save_plot=save_plot
                )
            if save_plot:
                fig_hist.write_image("%s_%s_hist.png" % (var_nm, target_nm))
        if return_plotly_object:
            return [fig, fig_hist]
        return res


@_time_decorator(
    time_info={
        "index": 2,
        "name": "time",
    },
    mode="column"
)
def woe_stab(
        var,
        Nvar,
        time,
        bad,
        varname='var',
        target='target',
        period=None,
        bins_method='quantile',
        avoid_null_mode=0,
        show_plot=True,
        save_plot=False,
        verbose=True,
        show_nulls=True,
        warnings_method='print',
        error_method='print',
        fontsize=18,
        path='',
):
    """
    Функция бьет переменную на бакеты и считает помесячно WoE, badrate и стабильность

    Parameters
    ----------
    var : array-like
        массив с переменной
    Nvar : int
        кол-во бакетов, на которое нужно разбить переменную
    time : array-like
        массив с переменной времени
    bad : array-like
        массив с целевой переменной
    transform_dt : bool, default False
        преобразовывать ли автоматически время в pd.datetime
        (в коде его нет, аргумент из декоратора)
    varname: str, default 'var'
        строка с названием переменной
    target: int, default 'target'
        целевая переменная
    period: str, default None
        один из коротких названий периодов, до которого нужно округлить значения в переменной времени
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    bins_method : str, default 'quantile'
        'quantile' или 'dense'
    avoid_null_mode : bool, default False
        флаг добавления 0.5 таргета в те бакеты, где он ни разу не принимает значение 1
    show_plot : bool, default True
        флаг отображения графика
    show_nulls: bool, default False
        флаг отображения бакета с пропущенными значениями
    verbose: bool, default True
        флаг подробного вывода информации по прогрессу
    show_nulls: bool, default True
        флаг демонстрации отсутствующих значений в массиве
    warnings_method: str, default 'print'
        функция, использующаяся для вывода предупреждений
    path: str, default ''
        путь до директории, в которую будет сохранена картинка

    Returns
    ---
    IV_time_all : list
        Список нужных нам метрик. Первым элементом выступает WoE,
        вторым badrate и третьим стабильность
    """

    if period is not None:
        arr_time = pd.to_datetime(time.dt.to_period(period).dt.to_timestamp())
    else:
        arr_time = time

    arr = var

    # обработка корректности входа
    if not isinstance(Nvar, int):
        raise IOError("Wrong 2nd argument")
    # обработка корректности данных столбцов
    if not (time.dtype == np.dtype('datetime64[ns]')):
        raise IOError("3rd argument has incorrect type")
    if not (set(bad.unique()) <= {math.nan, 0, 1}):
        raise IOError("4th argument has incorrect type")
    # потому что с дата-фреймом удобнее
    df_tmp = pd.DataFrame(data={varname: np.array(arr), 'time': np.array(arr_time), 'bad': np.array(bad)})

    if show_nulls:
        df_tmp = df_tmp.fillna(np.nan)
    else:
        n_with_nan = len(df_tmp)
        df_tmp = df_tmp.dropna()
        if n_with_nan != len(df_tmp):
            warnings.warn(f'Удалено {n_with_nan - len(df_tmp)} строк с нуллами, осталось {len(df_tmp)}', Warning,
                          stacklevel=2)

    input_data = df_tmp

    while Nvar >= 1:
        try:
            # Данные с бакетами
            input_data = input_data.assign(bucket=lambda x: _make_bucket(x[varname], Nvar, method=bins_method),
                                           obj_count=1)
            break
        except (IndexError, ValueError):
            # если не получилось побить на указанное число бакетов, бьем на меньшее число
            print(f'n_buckets {Nvar} reduced to {Nvar - 1}')
            Nvar -= 1

    unique_buckets = input_data['bucket'].unique()
    ####
    unique_buckets = unique_buckets[input_data['bucket'].value_counts() / input_data.shape[0] > 0.015]
    input_data = input_data[input_data['bucket'].isin(tuple(i for i in unique_buckets))]

    # выделение бакетов
    new_data = input_data.sort_values(by=varname, ascending=True)
    var_column = new_data[varname].values  # список значений целевой переменной
    quantity = np.empty(len(unique_buckets))  # массив числа переменных в бакете
    my_buckets = []  # список датафреймов по бакетам (list)
    my_buckets_value = []  # список

    for x in unique_buckets:
        my_buckets.append(new_data.loc[new_data['bucket'] == x])
        my_buckets_value.append(x)
    for i in range(len(my_buckets)):
        quantity[i] = len(my_buckets[i])
    number = np.empty(len(unique_buckets))
    for i in range(len(my_buckets)):
        number[i] = i + 1

    # работа со временем
    unique_time = list(set(new_data['time'].values))
    unique_time.sort()  # отсортированные уникальные значения времени (list)

    Nobs = 0  # общее число наблюдений
    Nbad = 0  # общее число плохих
    WoE_times = []  # список массивов значений WoE по времени для бакетов (list)
    WoE_lows = []  # список массивов значений нижней границы доверительного интервала WoE по времени для бакетов (list)
    bad_lows = []  # список массивов значений нижней границы доверительного интервала bad_rate по времени для бакетов (list)
    WoE_highs = []  # список массивов значений верхней границы доверительного интервала WoE по времени для бакетов (list)
    bad_highs = []  # список массивов значений верхней границы доверительного интервала bad_rate по времени для бакетов (list)
    bad_times = []  # список массивов значений bad_rate по времени для бакетов (list)
    part_times = []  # список массивов значений пропорций по времени для бакетов (list)
    IV_time_all = np.empty(len(unique_time))  # IV на всех
    Nobs_time = np.empty((len(my_buckets), len(unique_time)))  # количество наблюдений по бакетам и по времени
    Nbad_time = np.empty((len(my_buckets), len(unique_time)))  # количество плохих по бакетам и по времени
    Nobs_time_all = np.zeros(len(unique_time))  # количество в момент времени
    Nbad_time_all = np.zeros(len(unique_time))  # количество плохих в момент времени
    WoE_time_all = np.empty(len(unique_time))  # поправка в момент времени

    my_legend_old = []  # старая легенда для графиков
    for i in list(my_buckets_value):
        my_legend_old.append('var' + ' ' + 'in' + ' ' + i)

    for i in range(len(my_buckets)):  # идём по бакетам
        for j in range(len(unique_time)):  # идём по времени
            Nobs_time[i][j] = my_buckets[i][my_buckets[i]['time'] == unique_time[j]]['bad'].count()
            Nbad_time[i][j] = my_buckets[i][my_buckets[i]['time'] == unique_time[j]]['bad'].sum()

            Nobs_time[i][j], Nbad_time[i][j] = _check_bads_count(Nobs_time[i][j], Nbad_time[i][j], avoid_null_mode,
                                                                 str(unique_time[j]) + ' ' + my_legend_old[i],
                                                                 method=warnings_method)
            Nobs_time_all[j] += Nobs_time[i][j]
            Nbad_time_all[j] += Nbad_time[i][j]

    for j in range(len(unique_time)):
        Nobs_time_all[j], Nbad_time_all[j] = _check_bads_count(Nobs_time_all[j], Nbad_time_all[j], avoid_null_mode,
                                                               unique_time[j], method=warnings_method)

        Nobs += Nobs_time_all[j]
        Nbad += Nbad_time_all[j]
        WoE_time_all[j] = - math.log(Nbad_time_all[j] / (Nobs_time_all[j] - Nbad_time_all[j]))  # Считаем WoE по времени

        IV_time_all[j] = 0

    for i in range(len(my_buckets)):  # идём по бакетам
        WoE_time = np.empty(len(unique_time))  # массив значений WoE по времени для одного бакета (nparray)
        WoE_low = np.empty(len(
            unique_time))  # массив значений нижней границы доверительного интервала WoE по времени для одного бакета (nparray)
        WoE_high = np.empty(len(
            unique_time))  # массив значений верхней границы доверительного интервала WoE по времени для одного бакета (nparray)
        bad_low = np.empty(len(
            unique_time))  # массив значений нижней границы доверительного интервала bad_rate по времени для одного бакета (nparray)
        bad_high = np.empty(len(
            unique_time))  # массив значений верхней границы доверительного интервала bad_rate по времени для одного бакета (nparray)
        bad_time = np.empty(len(unique_time))  # массив значений bad_rate по времени для одного бакета (nparray)
        part_time = np.empty(len(unique_time))  # массив значений пропорций по времени для одного бакета (nparray)
        for j in range(len(unique_time)):  # идём по времени
            bad_time[j] = Nbad_time[i][j] / Nobs_time[i][j]
            part_time[j] = Nobs_time[i][j] / Nobs_time_all[j]

            WoE_time[j] = math.log(Nbad_time[i][j] / (Nobs_time[i][j] - Nbad_time[i][j])) + WoE_time_all[
                j]  # Считаем WoE по бакетам и по времени

            WoE_low[j] = math.log(
                _clopper_pearson(Nbad_time[i][j], Nobs_time[i][j])[0] / (
                        1 - _clopper_pearson(Nbad_time[i][j], Nobs_time[i][j])[0])) + \
                         WoE_time_all[j]
            WoE_high[j] = math.log(
                _clopper_pearson(Nbad_time[i][j], Nobs_time[i][j])[1] / (
                        1 - _clopper_pearson(Nbad_time[i][j], Nobs_time[i][j])[1])) + \
                          WoE_time_all[j]
            bad_low[j] = _clopper_pearson(Nbad_time[i][j], Nobs_time[i][j])[0]
            bad_high[j] = _clopper_pearson(Nbad_time[i][j], Nobs_time[i][j])[1]

            IV_time_all[j] = IV_time_all[j] + (
                    Nbad_time[i][j] / Nbad_time_all[j] - (Nobs_time[i][j] - Nbad_time[i][j]) / (
                    Nobs_time_all[j] - Nbad_time_all[j])) * WoE_time[j]

        WoE_lows.append(WoE_low)
        WoE_highs.append(WoE_high)
        bad_lows.append(bad_low)
        bad_highs.append(bad_high)
        WoE_times.append(WoE_time)
        part_times.append(part_time)
        bad_times.append(bad_time)

    bad_rate = Nbad / Nobs  # общий bad_rate

    if not (bad_rate == 0):
        WoE_All = -math.log(Nbad / Nobs / (1 - Nbad / Nobs))  # считаем общее WoE

    # вывод на экран

    if show_plot:

        WoE_avg = np.empty(len(my_buckets))
        for i in range(len(my_buckets)):
            WoE_avg[i] = WoE_times[i].mean()

        # пузырёк
        for i in range(0, len(WoE_avg)):
            for j in range(0, len(WoE_avg) - 1):
                if WoE_avg[j] > WoE_avg[j + 1]:
                    tmp_r = WoE_avg[j]
                    WoE_avg[j] = WoE_avg[j + 1]
                    WoE_avg[j + 1] = tmp_r
                    tmp_r = my_buckets[j]
                    my_buckets[j] = my_buckets[j + 1]
                    my_buckets[j + 1] = tmp_r
                    tmp_r = my_buckets_value[j]
                    my_buckets_value[j] = my_buckets_value[j + 1]
                    my_buckets_value[j + 1] = tmp_r
                    tmp_r = WoE_times[j]
                    WoE_times[j] = WoE_times[j + 1]
                    WoE_times[j + 1] = tmp_r
                    tmp_r = WoE_highs[j]
                    WoE_highs[j] = WoE_highs[j + 1]
                    WoE_highs[j + 1] = tmp_r
                    tmp_r = WoE_lows[j]
                    WoE_lows[j] = WoE_lows[j + 1]
                    WoE_lows[j + 1] = tmp_r
                    tmp_r = bad_times[j]
                    bad_times[j] = bad_times[j + 1]
                    bad_times[j + 1] = tmp_r
                    tmp_r = bad_highs[j]
                    bad_highs[j] = bad_highs[j + 1]
                    bad_highs[j + 1] = tmp_r
                    tmp_r = bad_lows[j]
                    bad_lows[j] = bad_lows[j + 1]
                    bad_lows[j + 1] = tmp_r
                    tmp_r = part_times[j]
                    part_times[j] = part_times[j + 1]
                    part_times[j + 1] = tmp_r

        my_legend = []  # новая легенда для графиков
        for i in range(0, len(my_buckets_value)):
            my_legend.append('var' + ' ' + 'in' + ' ' + my_buckets_value[i])

        nrows = 1
        ncols = 2
        figsize = (20, 8)

        legend_size = fontsize
        axes_size = int(fontsize // 1.5)

        fig, (ax1, ax4) = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(0, len(my_buckets)):
            ax1.plot(unique_time, list(WoE_times[i]))
            ax1.fill_between(unique_time, list(WoE_lows[i]), list(WoE_highs[i]), alpha=0.2)
        ax1.set_title("Зависимость WoE по бакетам от времени (%s, %s)" % (varname, target), fontsize=fontsize)
        ax1.legend(my_legend, loc='best', prop={'size': legend_size}, bbox_to_anchor=(0.45, -0.05))
        ax1.tick_params(labelsize=axes_size)
        ax1.grid()

        ax4.stackplot(unique_time, *part_times)
        ax4.legend(my_legend, loc='best', prop={'size': legend_size}, bbox_to_anchor=(0.45, -0.05))
        ax4.set_title("Распределение по бакетам от времени (%s, %s)" % (varname, target), fontsize=fontsize)
        ax4.tick_params(labelsize=axes_size)
        fig.tight_layout()

        if save_plot:
            fig.savefig(path + "%s_%s_14.png" % (varname, target))
            plt.close(fig)
        else:
            plt.show()

        fig, (ax3, ax2) = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(0, len(my_buckets)):
            ax3.plot(unique_time, list(bad_times[i]))
            ax3.fill_between(unique_time, list(bad_lows[i]), list(bad_highs[i]), alpha=0.2)
        ax3.set_title("Bad rate по бакетам от времени (%s, %s)" % (varname, target), fontsize=fontsize)
        ax3.legend(my_legend, loc='best', prop={'size': legend_size}, bbox_to_anchor=(0.45, -0.05))
        ax3.tick_params(labelsize=axes_size)
        ax3.grid()

        for i in range(0, len(my_buckets)):
            ax2.plot(unique_time, IV_time_all)
        ax2.set_title("Зависимость IV от времени (%s, %s)" % (varname, target), fontsize=fontsize)
        ax2.set_ylim([0, IV_time_all.max()])
        ax2.tick_params(labelsize=axes_size)
        ax2.grid()
        fig.tight_layout()

        if save_plot:
            fig.savefig(path + "%s_%s_32.png" % (varname, target))
            plt.close(fig)
        else:
            plt.show()

    return IV_time_all


def _get_IV(df, feature, target):
    """
    ...

    Parameters
    ---
    df : pandas.DataFrame
        Датафрейм с признаками
    feature : pandas.Series
        Предсказанное значение переменной
    target : pandas.Series
        Истинное значение перменной


    Returns
    ---
    data : pandas.DataFrame
        Датафрейм возвращаемых долей
    """
    lst = []

    # optional
    # df[feature] = df[feature].fillna("NULL")

    unique_values = df[feature].unique()
    for val in unique_values:
        lst.append([feature,  # Feature name
                    val,  # Value of a feature (unique)
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]  # Bad  (Fraud == 1)
                    ])

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'Good', 'Bad'])

    total_bad = df[df[target] == 1].count()[feature]
    total_good = df.shape[0] - total_bad

    data['Distribution Good'] = data['Good'] / total_good
    data['Distribution Bad'] = data['Bad'] / total_bad
    data['WoE'] = -np.log(data['Distribution Good'] / data['Distribution Bad']) # заменил

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    return data


def _make_df_stat_crosstab(df, name, nbuck):
    """
    Notes
    ---
    Техническая функция, вызывается внутри risks_crosstab
    """
    while nbuck >= 1:
        try:
            bucket = _make_bucket(df[name], num_buck=nbuck)
            break
        except (ValueError, IndexError):
            print(f'nbuck={nbuck} for {name} reduced to {nbuck - 1}')
            nbuck -= 1

    df_tmp = df.assign(bucket=bucket, obj_count=1)
    df_stat = df_tmp.groupby('bucket', as_index=False).agg(
        {'bad': 'sum', 'obj_count': 'sum', name: 'mean'}
    ).rename(columns={'bad': 'bad_count'})
    df[name] = bucket

    return df_stat, nbuck


def _process_labs(labels: list, accuracy_var: int) -> list:
    """
    Техническая функция для округления подписей на осях.
    Отдельно обрабатывает случай округления до целого (accuracy_var=0)
    и наличие бакета с нуллами.
    """

    for label in labels:
        text = label.get_text()
        text = float(text)

        if np.isinf(text):  # да-да-да, костыль для миссингов
            text = "missing"
        elif accuracy_var == 0:
            text = str(int(text))
        else:
            text = str(round(text, accuracy_var))

        label.set_text(str(text))

    return labels


def crosstab(bad,
           var_1, var_2,
           target='d4p12', name_1='feature_1', name_2='feature_2',
           null_buck_1=True, null_buck_2=True,
           accuracy=2, accuracy_var1=2, accuracy_var2=2,
           nbuck_1=8,
           nbuck_2=8,
           median=True,
           is_categorical_1=False,
           is_categorical_2=False,
           is_categorical_target=True,
           figsize=(7, 8.9),
           dpi=100,
           save_plot: bool = False,
           avoid_null_mode=0,
           min_badrate_bucket=0,
           min_samples_bucket=1,
           return_fig_axes=False,
           vmin_ratio=None,
           vmin_cnt=None,
           cnt_y_axis=True,
           to_clipboard=False
           ):
    """
    Функция разбивает две переменные на бакеты, и для каждой пары считает кол-во наблюдений и bad rate

    Parameters
    ---
    bad : array
        массив нулей и единиц, целевая переменная
    var_1 : array
        массив значений первой переменной
    var_2 : array
        массив значений второй переменной
    target : str, default 'd4p12'
        название таргета
    name_1 : str, default 'feature_1'
        название первой переменной для подписей к графикам
    name_2 : str, default 'feature_2'
        название второй переменной для подписей к графикам
    null_buck_1 : bool, default True
        при значении True включает бакет с пустыми значениями 1-ой переменной, при False - исключает
    null_buck_2 : bool, default True
        при значении True включает бакет с пустыми значениями 1-ой переменной, при False - исключает
    accuracy : int, default 2
        количество знаков после запятой (в процентах) в таблице с bad_rate
    accuracy_var1 : array-list, default 2
        аналогично для первой переменной
    accuracy_var2 : array-list, default 2
        аналогично для второй переменной
    nbuck_1 : int, default 8
        количество бакетов для первой переменной
    nbuck_2 : int, default 8
        количество бакетов для второй переменной
    median : bool, default True
        при значении True отображает вместо границ бакетов медианное значение в них
    is_categorical_1 : bool, default False
        флаг категориальной первой переменной
    is_categorical_2 : bool, default False
        флаг категориальной второй переменной
    is_categorical_target : bool, default True
        флаг категориального таргета
    figsize : tuple, default (7, 8.9)
        (широта, высота) графика
    dpi : int, default 100
        dpi (четкость) картинки
    save_plot : bool, default False
        при значении True сохраняет график. При этом график не отрисовывается и fig, axes не возвращаются,
        даже если выбрано return_fig_axes=True.
    avoid_null_mode : bool, default 0
        флаг, 1, [если хотите в бакеты без bad добавлять 1 bad (и ломать этим реальную картину и здравый смысл)(с) - Влад], 0 - если нет
    min_badrate_bucket : float, default 0.001
        Бэдрейт, начиная c которого начинает работать цветовая схема кросс-таба. Для непрерывного таргета -- минимальное
        среднее значение таргета в бакете, для которого работает цветовая схема.

        Ограничение работает **одновременно** с `min_samples_bucket`. Например, если `min_badrate_bucket=0.01` и
        `min_samples_bucket=100`, отрисовываться будут только бакеты с бедрейтом не менее 1% **и** с не менее
        100 наблюдениями. Ограничение **общее** для верхнего и нижнего графиков.
    min_samples_bucket : int, default 1
        Количество наблюдений, начиная c которых начинает работать цветовая схема кросс-таба.

        Ограничение работает **одновременно** с `min_badrate_bucket`. Например, если `min_badrate_bucket=0.01` и
        `min_samples_bucket=100`, отрисовываться будут только бакеты с бедрейтом не менее 1% **и** с не менее
        100 наблюдениями. Ограничение **общее** для верхнего и нижнего графиков.
    return_fig_axes : bool, default False
        Если True, функция вернет объекты `fig`, `ax` (которые возвращает plt.subplots). Это можно использовать, чтобы
        провести более тонкую настройку графика. Пример работы:

            fig, axes = risks_crosstab(..., return_fig_axes=True)

            # поворачиваем и настраиваем шрифт меток на осях
            for ax in axes:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=10)

            # большое название всего графика
            fig.suptitle("Big title", y=1.01)

    vmin_ratio: float, default None
        Аргумент **не используется**, оставлен для обратной совместимости. Если указать значение,
        функция напечатает предупреждение.
    vmin_cnt: float, default None
        Аргумент **не используется**, оставлен для обратной совместимости. Если указать значение,
        функция напечатает предупреждение.
    cnt_y_axis : bool, default True
        флаг отображения подписи к оси Y на графике с числом значений в бакетах
    to_clipboard : bool, default False
        если True, полученный кросстаб копируется в буфер обмена, и его можно вставить в excel.

    Notes
    ----
    Под капотом функция дропает наны в таргете
    """
    # нерабочие комбинации аргументов
    if median and (is_categorical_1 or is_categorical_2):
        raise ValueError((
            "median=True is not supported for categorical variables.\n"
            "Choose median=False or both is_categorical_1=False "
            "and is_categorical_2=False to avoid this error.\n"
        ))

    if vmin_ratio or vmin_cnt:
        print((
            "Args vmin_ratio and vmin_cnt are no longer supported.\n"
            "Use min_badrate_bucket / min_samples_bucket instead.\n"
        ))

    if save_plot and return_fig_axes:
        print((
            "If save_plot=True, (fig, axes) will not be returned.\n"
        ))

    # вспомогательный датафрейм и обработка нуллов
    df = pd.DataFrame({
        name_1: np.array(var_1),
        name_2: np.array(var_2),
        'bad': np.array(bad)
    })

    target_nans = df['bad'].isna()
    target_nans_count = target_nans.sum()
    if target_nans_count > 0:
        df = df.loc[~target_nans]
        print(f"{target_nans_count:g} rows with missing target were dropped.")

    if not null_buck_1:
        df = df.dropna(subset=[name_1])
    if not null_buck_2:
        df = df.dropna(subset=[name_2])

    # разбиваем переменные на бакеты и считаем по бакетам агрегаты
    if not is_categorical_1:
        df_stat_var_1, nbuck_1 = _make_df_stat_crosstab(df, name_1, nbuck_1)
    else:
        nbuck_1 = df[name_1].nunique()

    if not is_categorical_2:
        df_stat_var_2, nbuck_2 = _make_df_stat_crosstab(df, name_2, nbuck_2)
    else:
        nbuck_2 = df[name_2].nunique()

    df["cnt"] = 1

    # матрицы для отрисовки
    if median:
        name_1_y, name_2_y = f"{name_1}_y", f"{name_2}_y"
        df = df.merge(df_stat_var_1, left_on=name_1, right_on='bucket')
        df = df.merge(df_stat_var_2, left_on=name_2, right_on='bucket')

        # костыль, чтобы нулловый бакет был обработан pd.pivot_table
        df[name_1_y] = df[name_1_y].fillna(np.inf)
        df[name_2_y] = df[name_2_y].fillna(np.inf)

    table1_for_heatmap = pd.pivot_table(
        df, values='bad',
        index=name_1_y if median else name_1,
        columns=name_2_y if median else name_2,
        aggfunc=np.mean
    )

    table2_for_heatmap = pd.pivot_table(
        df, values='cnt',
        index=name_1_y if median else name_1,
        columns=name_2_y if median else name_2,
        aggfunc=np.sum,
        fill_value=0  # чтобы при наличии нуллов int не превращался во float
    )

    fig, axes = plt.subplots(2, 1, sharey=True, figsize=figsize, dpi=dpi)
    # mask -- где НЕ отрисовываем
    mask = (table2_for_heatmap < min_samples_bucket) | (table1_for_heatmap < min_badrate_bucket)
    fmt = f".{accuracy}{'%' if is_categorical_target else 'f'}"
    cmap = mcolors.LinearSegmentedColormap.from_list('gyr', ['green', 'yellow', 'red'])

    # верхний график
    sns.heatmap(
        table1_for_heatmap,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        mask=mask,
        ax=axes[0]
    )

    # нижний график
    sns.heatmap(
        table2_for_heatmap,
        annot=True,
        fmt='g',
        cmap='Greys',
        mask=mask,
        ax=axes[1]
    )

    if to_clipboard:
        # пустая строка между двумя датафреймами
        n = len(table1_for_heatmap.columns)
        empty_row = pd.DataFrame(data=np.full((1, n), None), columns=table1_for_heatmap.columns, index=[None])

        # приводим датафреймы к такому же виду, в котором их рисует sns.heatmap
        table1_for_heatmap = table1_for_heatmap.applymap(
            f"{{:{fmt}}}".format)  # пример того, что получается в строке: '{:.2%}'
        table1_for_heatmap[mask] = None
        table2_for_heatmap[mask] = None

        # соединяем и копируем в буфер обмена
        df_to_clipboard = pd.concat([table1_for_heatmap, empty_row, table2_for_heatmap])
        df_to_clipboard.to_clipboard()
        print("Crosstab has been copied to clipboard. Now you can paste it into an excel file.")

    if median:
        # округляем подписи на картинках
        xlab, ylab = axes[0].get_xticklabels(), axes[0].get_yticklabels()
        xlab, ylab = _process_labs(xlab, accuracy_var1), _process_labs(ylab, accuracy_var2)

        # обновляем подписи
        axes[0].set_xticklabels(xlab, rotation=0), axes[1].set_xticklabels(xlab, rotation=0)
        axes[0].set_yticklabels(ylab, rotation=0), axes[1].set_yticklabels(ylab, rotation=0)

    for ax in axes:
        ax.set_xlabel(name_2, fontsize=14)
        ax.set_ylabel(name_1, fontsize=14)

    if not cnt_y_axis:
        plt.setp(axes[1].get_yticklabels(), visible=False)

    # сохраняем
    if save_plot:
        fig.tight_layout()
        fig.savefig(f"{name_1}_{name_2}_{target}_cross.png")
        plt.close()

    elif return_fig_axes:
        return fig, axes

    else:
        plt.show()


def _stability_index_buckets(actual: np.ndarray, expected: np.ndarray) -> np.float64:
    """
    Функция для рассчета SI по числу элементов в n бакетах

    actual: np.ndarray - массив длины n с числом элементов по бакетам в новой выборке
    expected: np.ndarray - массив длины n с числом элементов по бакетам в старой выборке

    Notes
    -----
    Технические функции
    """
    actual_prob = actual / actual.sum()
    expected_prob = expected / expected.sum()
    return np.sum((actual_prob - expected_prob) * np.log(actual_prob / expected_prob))


@_time_decorator(
    time_info={
        "index": 3,
        "name": "time",
        "df_index": 0,
        "df_name": "df"
    },
    mode="column_nm"
)
def plot_metric_time(df,
                 features,
                 target='d4p12',
                 time='month_dt',
                 metric=roc_auc_score,
                 metric_name='roc auc',
                 verbose=False,
                 show_plot=True,
                 interactive_plot=False,
                 save_plot=False,
                 ttl=None,
                 last_month_of_the_train=None,
                 segmentation_rule=None,
                 ttl_size=12,
                 legend_size=12,
                 plot_size=(12, 5),
                 linewidth=2,
                 ticks_rotation=15,
                 ylim=(None, None),
                 return_df=True,
                 sample_size_plot=False,
                 bin_color='k',
                 bin_alpha=0.2,
                 bin_width=25,
                 bin_height=0.5,
                 ):
    """
    Отрисовываем метрику во времени для нескольких скоров во времени.
    Возвращает словарь, при return_dic=True, с ключами названиями скоров features, и значениями - массивами 2 на N,
        где первый подмассив - значения метрики на конкретроном месяце для данного скора
            второй - месяц, на котором считалась метрика

    Для roc_auc_score:
    Функция автоматически меняет auc < 0.5 на 1 - auc. Т.е. результат 0.53 может означать в том числе и 0.47,
        а 0.8 может быть 0.2 до перестановки классов.

    Parameters
    ----------
    df : pd.DataFrame
        df с данными
    features : list
        список с названиями скоров
    target : str, default 'd4p12'
        название таргета, колонка df
    time : str, default 'month_dt'
        название колонки с датой
        Дата должна быть приведена к datetime
        Например, так

            df['month_dt'] = pd.to_datetime(df['month_dt'])

    metric : function, default sklearn.metrics.roc_auc_score
        функция метрики, для которой строится график
    metric_name : str, default 'roc auc'
        название метрики
    transform_dt : bool, default False
        преобразовывать ли автоматически время в pd.datetime
        (в коде его нет, аргумент из декоратора)
    verbose : bool, default False
        аргумент для обратной совместимости
        Ни на что не влияет
    show_plot : bool, default True
        рисовать ли график
    interactive_plot : bool, default False
        аргумент для обратной совместимости
        Ни на что не влияет
    save_plot : bool, default False
        сохранить ли график
    ttl : str, default None
        название графика
    last_month_of_the_train : pandas._libs.tslibs.timestamps.Timestamp, default False
        последний месяц трейна.
        Отрисует вертикальную черту, обозначающую конец обучающего периода
        По дефолту заполняется None.
        Тип datetime, Timestamt.
        Например:
            pd.Timestamp('2020-01-01')
    segmentation_rule : np.array, default False
        ограничения на dataframe, записывается в виде маски.
        По дефолту заполняется None.
        Например, df.d4p12.notnull().
    ttl_size : int, default 12
        размер шрифта в названии графика
    legend_size : int, default 12
        размер шрифта в легенде графика
    plot_size : int, default (12, 5)
        размер графика
    linewidth : int, default 5
        толщина линий
    ticks_rotation : numeric, default 15
        угол поворота xticklabels
    ylim : tuple, default (None, None)
        верхняя и нижняя граница графика по y. Можно указать обе
        а можно – только верхнюю или нижнюю. Для этого указываем
        в для другой границы None, и она расчитается автоматически.
    return_df : bool, default True
        возвращать ли датафрейм с рок ауками по месяцам
    conf_int : list, default []
        список переменных, для которых строим доверительный интервал для roc auc
    conf_int_alpha : float, default 0.2
        прозрачность доверительного интервала
    sample_size_plot : bool, default False
        флаг отображения столбиков с размерами выборки
    bin_alpha : float, default 0.2
        прозрачность столбиков
    bin_width : int/float, default 25
        ширина столбиков, может потребоваться настройка
        в зависимости от периода отрисовки
    bin_height : float, default 0.5
        высота самого высокого столбца в долях от высоты оси y
    """

    if segmentation_rule is None:
        segmentation_rule = [True]*df.shape[0] # Костыль, чтобы не брать подмножество датафрейма

    if df[segmentation_rule][features].isna().sum().sum() > 0:
        raise ValueError('Input contains NaN !!!!')

    if len(features) > 10:
        raise OSError('too many features, 10 is the maximum')

    dict_metric = {i: [] for i in features}
    months = df[time].unique()
    months.sort()

    for score in dict_metric.keys():
        tmp1 = []
        tmp2 = []
        ft_metric = metric(df[segmentation_rule & df[score].notnull()][target],
                           df[segmentation_rule & df[score].notnull()][score])
        for j in months:
            segmentation_rule1 = segmentation_rule & (df[time] == j) & df[score].notnull()
            if (df[segmentation_rule1].shape[0] > 0) & \
                    (df[segmentation_rule1][target].nunique() > 1):
                tmp = metric(df[segmentation_rule1][target],
                             df[segmentation_rule1][score])

                tmp1.append(tmp)
                tmp2.append(j)
        # Если строим roc_auc во времени
        if metric == roc_auc_score and ft_metric < 0.5:
            tmp1 = list(1 - np.array(tmp1))
            warnings.warn(
                f'Скор {score} работает в другую сторону!', Warning, stacklevel=2)
        dict_metric[score] = [tmp1, tmp2]

    plt.figure(figsize=plot_size)

    color_counter = 0
    for i in dict_metric.keys():
        tmp1 = pd.DataFrame(dict_metric[i][0], columns=[metric_name])
        tmp2 = pd.DataFrame(dict_metric[i][1], columns=[time])
        tmp = pd.concat([tmp1, tmp2], axis=1)

        plt.plot(tmp[time].values,
                 tmp[metric_name].values,
                 '-o',
                 label=i,
                 linewidth=linewidth,
                 alpha=0.6)  # прозрачность линий при наложении
        color_counter += 1

    # Меняем границы графика по оси Оy
    plt.ylim(bottom=ylim[0], top=ylim[1])
    # Берем границы графика по оси Oy для построения линии, разделяющей трейн и тест
    min_ylim, max_ylim = plt.ylim()

    if last_month_of_the_train is not None:
        plt.vlines(
            last_month_of_the_train,
            min_ylim,
            max_ylim,
            colors='r',
            label='end of train',
            linewidth=linewidth
        )
    plt.legend(prop={'size': legend_size})

    if ttl:
        plt.title(ttl, size=ttl_size)

    plt.grid()
    plt.xticks(rotation=ticks_rotation)

    plt.ylabel(metric_name)

    if sample_size_plot:
        ax_plot_size = plt.gca().twinx()
        sample_sizes = df[segmentation_rule].groupby(time)[target].count()
        ax_plot_size.bar(
            x=sample_sizes.index,
            height=sample_sizes,
            color=bin_color,
            alpha=bin_alpha,
            width=bin_width
        )
        ax_plot_size.set(
            ylabel='sample size',
            ylim=(0, np.max(sample_sizes) / bin_height),
        )
        ax_plot_size.grid(False)

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"{metric_name}_time_{target}.png")
    if show_plot:
        plt.show()
    else:
        plt.close(plt.gcf())

    if return_df:
        # может получиться так, что разные скоры посчитаны для разных моментов времени
        # поэтому сначала соберем все уникальные моменты, а к ним притянем скоры, если они существуют
        time_index = np.unique(np.array([dict_metric[key][1] for key in dict_metric.keys()]))
        feature_results = []

        for key in dict_metric.keys():
            auc_values, time_moments = dict_metric[key][0], dict_metric[key][1]
            feature_results.append([
                auc_values[time_moments.index(moment)] if moment in time_moments else None for moment in time_index
            ])

        return pd.DataFrame(dict(zip(features, feature_results)), index=time_index)


def _auc_stab(arr):
    """
    Вычисление метрики стабильности по списку auc в разные периоды
    arr - array like: список значений (например, помесячных)

    Notes
    -----
    Техническая функция
    """
    mean = arr.mean() - 0.5
    ratios = (arr - 0.5) / mean

    log = np.log(ratios)

    res = log.mean()

    return np.power(np.exp(res), 5)


def _iv_stab(arr):
    """
    Notes
    -----
    Техническая функция
    """
    mean = arr.mean()
    ratios = arr / mean

    log = np.log(ratios)

    res = log.mean()
    return np.power(np.exp(res), 5)


class WoEFiller(object):
    """
    Класс с тремя методами, которые обрабатывают датафреймы с пустыми значениями переменных:
    fit()
    transform()
    fit_transform()
    """

    def __init__(self):
        self.fill_dict = {}

    def fit(self, X, y, manual_values=None, method='linear'):
        """
        Метод вычисляет значения, которыми нужно заполнять пустые значения (null-ы).
        Вычисляется данное значение с помощью логистической регрессии по этой переменной.

        Parameters
        ---
        X : pandas.DataFrame
            датафрейм с переменными
        y : array-like
            вектор таргета
        manual_values : dict, default None
            словарь, в котором для переменных можно указать значения вручную, чем заполнять null-ы
        method : str, default 'linear'
            как заполнять null-ы:

             - 'linear' - чтобы сохранить линейность WoE от бакетов переменной

             - 'zero_woe' - нулевым по woe значением

        Returns
        ---
        fill_dict : dict
            Словарь с переменными и подсчитанными для заполнения null'ов значениями.

        Notes
        ---
        Переменная должна быть линеаризована перед заполнением null-ов. Значение, которым
        заполняется null, может вылезти за границы переменной, не забудьте его ограничить после.
        """
        if method not in ('linear', 'zero_woe'):
            raise ValueError(f'Wrong method "{method}", try "linear" or "zero_woe"')
        if manual_values is None:
            manual_values = {}
        if type(X) not in (pd.DataFrame, pd.Series, list, tuple, set)\
           and (type(X) != np.ndarray or type(X) == np.ndarray and len(X.shape) in (1, 2)):
            raise TypeError("fit(X, y, manual_values): Type X is not suitable for Dataframe.")
        if type(y) not in (pd.DataFrame, pd.Series, list, tuple, set)\
           and (type(y) != np.ndarray or type(y) == np.ndarray and len(y.shape) in (1, 2)):
            raise TypeError("fit(X, y, manual_values): Type y is not suitable for Dataframe.")
        if manual_values is not None and type(manual_values) != dict:
            raise TypeError("fit(X, y, manual_values): manual_values is not dictionary.")
        if type(X) in (pd.Series, list, tuple, set, np.ndarray):
            X = pd.DataFrame(X)
        if type(y) in (list, tuple, set, np.ndarray):
            y = pd.DataFrame(y)
        if X.shape[0] != y.shape[0]:
            raise TypeError("fit(X, y, manual_values): X and y have a different number of rows.")
        self.fill_dict = {}

        if not (set(y) == {0, 1}):
            raise IOError(f"Wrong unique values for target columns: {set(y)}")

        df_tmp = X.assign(target=y)
        scaler = StandardScaler()
        clf = LogisticRegression(penalty='none', solver='lbfgs', max_iter=500)

        # Проходим по переменным, где есть null-значения и которых нет в manual_values.
        for column in df_tmp.columns[:-1]:
            if manual_values is None or manual_values.get(column) is None:
                if (df_tmp[df_tmp[column].notna()]['target'].nunique() == 1):
                    warnings.warn(
                        f'При ненулевых значениях столбца {column} таргет имеет только одно уникальное значение.'
                        f'Нуллы в столбце будут заполнены средним значением.', Warning, stacklevel=2)
                    mean_value = df_tmp[column].mean()
                    df_tmp[column] = df_tmp[column].fillna(mean_value)

                df_column = df_tmp.loc[:, (column, 'target')].dropna()

                scaler.fit(df_column[[column]])
                clf.fit(scaler.transform(df_column[[column]]), df_column['target'])

                nulls_target = df_tmp[df_tmp[column].isna()]['target'].values
                # если пропусков нет, заполняем нулем по WoE
                if len(nulls_target) == 0:
                    method = 'zero_woe'

                # учитываем woe нулла, заполняем так, чтобы сохранилась линейность
                if method == 'linear':
                    logodds_null = np.log(
                        (
                            len(nulls_target) - nulls_target.sum()
                        ) / nulls_target.sum()
                    )

                    norm_value = logodds_null

                # если нет нуллов в выборке или мы считаем, что его woe не значимо, заполняем нулевым WoE
                # рассчитанным по подвыборке, где переменная определена
                if method == 'zero_woe':
                    logodds_notnull_sample = np.log(
                        (
                            len(df_column['target']) - df_column['target'].sum()
                        ) / df_column['target'].sum()
                    )

                    norm_value = logodds_notnull_sample

                # norm_value - значение, к которому мы приравниваем логит. При norm_value = 0
                # value будет таким, что вероятность таргета в пропущенных значениях будет 1/2
                value = -((clf.intercept_[0] + norm_value) / clf.coef_[0][0]) * scaler.scale_ + scaler.mean_

                self.fill_dict.update({column: value[0]})

                # Если ключ из manual_values есть в X, добавляем его со значением в fill_dict
            elif manual_values is not None and manual_values.get(column) is not None:
                self.fill_dict.update({column: manual_values[column]})
        return self.fill_dict

    def transform(self, X):
        """
        Метод заполняет null-ы переменных значениями, подсчитанными в методе класса fit().

        Parameters
        ---
        X : pandas.DataFrame
            датафрейм с переменными

        Returns
        ---
        df_tmp : pandas.DataFrame
            Обновлённый DataFrame с заполненными null'ами и с добавленными переменными - флагами
            пустых значений для каждой переменной.

        Notes
        ---
        Заполняются null'ы и добавляются флаги пустых значений только тех переменных, которые
        оперировали при вызове метода класса fit().
        """

        df_tmp = X.copy()
        for column in self.fill_dict.keys():
            df_tmp[column + '_null_flg'] = df_tmp[column].isna()
            df_tmp[column] = df_tmp[column].fillna(self.fill_dict[column])
        df_tmp.replace({True: 1, False: 0}, inplace=True)
        df_tmp = df_tmp.loc[:, (df_tmp!=0).any(axis=0)]
        return df_tmp

    def fit_transform(self, X, y, manual_values=None, method='linear'):
        """
        Метод последовательно вызывает методы класса fit() и transform().
        """
        if manual_values is None:
            manual_values = {}
        self.fit(X, y, manual_values, method=method)
        return self.transform(X)


def _check_nan(df: pd.DataFrame, method='print') -> pd.DataFrame:
    """
    Проверят наличие нуллов в датафреме. При нахождении удалаяет с выводом предупреждения
    df -- датафрейм для проверки
    """
    n_with_nan = len(df)
    df = df.dropna()
    if n_with_nan != len(df):
        if method == 'print':
            print(f'Удалено {n_with_nan - len(df)} строк с нуллами, осталось {len(df)}')
        elif method == 'warning':
            warnings.warn(f'Удалено {n_with_nan - len(df)} строк с нуллами, осталось {len(df)}', Warning, stacklevel=2)
        elif method == 'silent':
            pass
        else:
            raise ValueError(f'Undefined method "{method}", use "print" or "warning"')
    return df


def _check_bads_count(n_obs, n_bad, avoid_null_mode, bucket, method='print'):
    if n_obs == 0 or n_bad == 0 or n_bad == n_obs:
        if method == 'print':
            print(f''' Wrong objects for {bucket}: total {n_obs}, target {n_bad}''')
        elif method == 'warning':
            warnings.warn(f''' Wrong objects for {bucket}: total {n_obs}, target {n_bad}''', Warning, stacklevel=2)
        elif method == 'silent':
            pass
        else:
            raise ValueError(f'Undefined method "{method}", use "print" or "warning"')

        if avoid_null_mode:
            n_bad = max(0.5, n_bad)
            n_obs = max(1.0, n_obs)
            n_bad = min(n_bad, n_obs - 0.75)
        else:
            raise ZeroDivisionError('Use avoid_null_mode = 1 to handle this error')

    return n_obs, n_bad


def _add_null_flags(X, has_nulls, flags_df):
    for col in X.columns:
        if col in has_nulls:
            flg = f"{col}_null_flg"
            X[flg] = flags_df[flg]
    return X


def forward_selection(
        Q, y,
        initial_list=None,
        threshold_in=0.01,
        null_mode='raise',
        verbose=True
):
    """
    Пошаговый отбор параметров на основе значимости.

    Parameters
    ---
    Q : pandas.DataFrame
        датафрейм с признаками для отбора
    y : array-like
        вектор таргета
    initial_list : array-like, default []
        лист с переменными, с которых начинается алгоритм
    threshold_in : float, default  0.01
        добавляем параметр, если p-value < ``threshold_in``
    null_mode : str, default 'raise'
        как обрабатывать пропуски:

         - 'raise' : никак, если они есть, функция упадет с ошибкой

         - 'auto' : автоматически добавить флаги пропусков

    verbose : int or bool, default True
        вывод подробной информации

         - verbose > 1 даст еще более подробный вывод

    Returns
    ---
    included : list
        Список отобранных переменных (их названия) в порядке важности.
        Вначале самая значимая, в конце наименее значимая. Флаги нуллов в список
        не входят. Если признак в списке, его нужно использовать
        вместе с его флагом пропуска.

    Notes
    ---
    Может так оказаться, что у двух фичей пропуски в одинаковых строках,
    что приведет к добавлению двух одинаковых флагов пропусков. В итоге,
    если оба признака отберутся, у матрицы признаков итоговой модели
    будет линейно зависимые столбцы.
    """

    # флаги пропущенных значений
    if null_mode == 'raise':
        has_nulls = []
        flags_df = pd.DataFrame([])

    if null_mode == 'auto':
        flags_df = Q.isna().astype(int)
        flags_df = flags_df.loc[:, flags_df.sum(0) > 0]
        has_nulls = flags_df.columns
        flags_df.columns = [col + '_null_flg' for col in flags_df.columns]

        if verbose > 0:
            print("Add null flags to:\n" + '\n'.join(has_nulls) + '\n')

    # Нормируем X
    if initial_list is None:
        initial_list = []
    included = list(initial_list)

    scaler = StandardScaler()
    scaler.fit(Q)
    X = pd.DataFrame(scaler.transform(Q), columns=Q.columns)

    if null_mode == 'auto':
        X.fillna(0, inplace=True)

    while True:
        changed = False

        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype='float64')
        new_logloss = pd.Series(index=excluded, dtype='float64')

        if len(included) == 0:
            model = LogisticRegression(penalty='none', fit_intercept=False, max_iter=500, solver='lbfgs')
            model.fit(np.full(shape=len(y), fill_value=1).reshape(-1, 1), y)
            R1 = log_loss(y, model.predict_proba(np.full(shape=len(y), fill_value=1).reshape(-1, 1))[:, 1]) * y.shape[0]

        else:
            X_current = X[included].copy()
            X_current = _add_null_flags(X_current, has_nulls, flags_df)

            model = LogisticRegression(penalty='none', max_iter=500)
            model.fit(X_current, y)
            R1 = log_loss(y, model.predict_proba(X_current)[:, 1]) * y.shape[0]

        maxim = 0
        for new_column in excluded:
            X_current = X[included + [new_column]].copy()
            X_current = _add_null_flags(X_current, has_nulls, flags_df)

            model_enter = LogisticRegression(penalty='none', max_iter=500).fit(X_current, y)
            R2 = log_loss(y, model_enter.predict_proba(X_current)[:, 1]) * y.shape[0]

            new_pval[new_column] = chi2.sf(2 * (R1 - R2), 1)
            new_logloss[new_column] = R2

            if verbose > 1:
                print('{:30} has p-value {:15.6}; Logloss(current) = {:5.7}, Logloss(after add) = {:.7}'.format(
                    new_column, new_pval[new_column], R1,
                    R2))
            if maxim < 2 * (R1 - R2):
                maxim = 2 * (R1 - R2)

        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_logloss.index[new_logloss.argmin()]
            included.append(best_feature)

            changed = True
            if verbose > 0:
                print('\033[1m' + 'Add {:30} with p-value {:.6}'.format(best_feature, best_pval) + '\033[0m')
        if not changed:
            break
    return included