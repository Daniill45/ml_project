
import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from datetime import datetime
from sqlalchemy import create_engine
from schema import PostGet
import re


def trans(X):
    # Создаем функцию, чтобы в модель попала корректная форма с пользователем
    X = pd.get_dummies(X, columns=['country', 'topic'])

    columns_dummies = ['country_Azerbaijan', 'country_Belarus', 'country_Cyprus',
                       'country_Estonia', 'country_Finland', 'country_Kazakhstan',
                       'country_Latvia', 'country_Russia', 'country_Switzerland',
                       'country_Turkey', 'country_Ukraine', 'topic_business', 'topic_covid',
                       'topic_entertainment', 'topic_movie', 'topic_politics', 'topic_sport',
                       'topic_tech']
    X[list(set(columns_dummies) - set(X.columns))] = 0
    return X[['user_id', 'gender', 'age', 'exp_group', 'os', 'source',
              'population', 'admin_name', 'is_capital', 'parity_gdp',
              'post_id', 'day_of_week', 'month', 'hour', 'year', 'day',
              'DistanceToCluster_0', 'DistanceToCluster_1',
              'DistanceToCluster_2', 'DistanceToCluster_3', 'DistanceToCluster_4',
              'DistanceToCluster_5', 'DistanceToCluster_6', 'DistanceToCluster_7',
              'DistanceToCluster_8', 'DistanceToCluster_9', 'DistanceToCluster_10',
              'DistanceToCluster_11', 'DistanceToCluster_12', 'DistanceToCluster_13',
              'DistanceToCluster_14', 'DistanceToCluster_15', 'DistanceToCluster_16',
              'DistanceToCluster_17', 'DistanceToCluster_18', 'DistanceToCluster_19',
              'DistanceToCluster_20', 'DistanceToCluster_21', 'DistanceToCluster_22',
              'DistanceToCluster_23', 'DistanceToCluster_24', 'mean_tf_idf_2',
              'max_tf_idf_2', 'lenght_text', 'country_Azerbaijan', 'country_Belarus', 'country_Cyprus',
              'country_Estonia', 'country_Finland', 'country_Kazakhstan',
              'country_Latvia', 'country_Russia', 'country_Switzerland',
              'country_Turkey', 'country_Ukraine', 'topic_business', 'topic_covid',
              'topic_entertainment', 'topic_movie', 'topic_politics', 'topic_sport',
              'topic_tech']]


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("my/super/path")
    from_file = CatBoostClassifier()
    loaded_model = from_file.load_model(model_path)

    return loaded_model


def batch_load_sql(query: str):
    engine = create_engine("postgresql://username:password@host:port/database")
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    liked_posts_query = """
    SELECT distinct post_id, user_id
    FROM public.feed_data
    WHERE action='like'
    """
    liked_posts = batch_load_sql(liked_posts_query)
    posts_features = pd.read_sql("""
    SELECT * FROM public.posts_info_features""",
                                 con="postgresql://username:password@host:port/database")
    user_features = pd.read_sql("""
    SELECT * FROM public.user_data""",
                                con="postgresql://username:password@host:port/database")

    return [liked_posts, posts_features, user_features]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\sa-zA-Z\d@\[\]]', ' ', text)  # Удаляет пунктцацию
    text = re.sub(r'\w*\d+\w*', '', text)  # Удаляет цифры
    text = re.sub(' ', "", text)  # Удаляет ненужные пробелы
    return text


def city_exist(x, a):
    if x in a:
        return x
    return "no_city"


def prepare_user(user_data):
    gdp = {'Russia': 32803, 'Ukraine': 14220, 'Belarus': 21699, 'Azerbaijan': 15843, 'Kazakhstan': 28600,
           'Finland': 53654, 'Turkey': 30472, 'Latvia': 34469, 'Cyprus': 42556, 'Switzerland': 77324, 'Estonia': 42192}
    world_cities = pd.read_csv("/yourpath/worldcities.csv")  # Датасет городов мира
    countries = list(user_data.country.unique())
    world_cities = world_cities[world_cities.country.apply(lambda x: x in countries)]  # Оставляем только нужные страны
    world_cities = world_cities[["city", "country", "population", "admin_name", "capital"]]
    user_data.city = user_data.city.apply(lambda x: clean_text(x))
    world_cities.city = world_cities.city.apply(lambda x: clean_text(x))
    a = set(world_cities.city)
    user_data.city = user_data.city.apply(
        lambda x: city_exist(x, a))  # функция вовращает есть ли такой город в списке или нет
    user_data = user_data[user_data.city != "no_city"]
    user_data.admin_name.fillna(user_data.city, inplace=True)
    user_data["is_capital"] = 0  # Живет ли человек в столице
    user_data.loc[user_data.capital == 'primary', "is_capital"] = 1
    user_data = user_data.drop(["capital"], axis=1)
    user_data = user_data.rename({"country_x": "country"}, axis=1)
    user_data["parity_gdp"] = 0
    user_data["parity_gdp"] = user_data.country.apply(lambda x: gdp[x])
    user_data["os"] = user_data.os.apply(lambda x: 1 if x == "iOS" else 0)
    user_data["source"] = user_data.source.apply(lambda x: 1 if x == "ads" else 0)
    return user_data


app = FastAPI()

model = load_models()
features = load_features()
engine = create_engine(
    "postgresql://robot-startml-ro:password@"
    "postgres.lab.karpov.courses:6432/startml"
)

new_post = pd.read_sql('SELECT * FROM posts_with_embeddings', con=engine)

user_data = pd.read_sql(
    """SELECT * FROM public.user_data """, con=engine
)
user_data = prepare_user(user_data)
world_cities = pd.read_csv("/yourpath/worldcities.csv")


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    # Создаем датафрейм для 1 го юзера на 7023 поста и будем проверять каждый, далее отсортируем по вероятности
    df_predict = pd.DataFrame({'post_id': new_post['post_id'], 'user_id': id})
    df_predict = df_predict.merge(user_data, how='left', left_on='user_id', right_on='user_id')
    df_predict = df_predict.merge(new_post, how='left', left_on='post_id', right_on='post_id')
    # Добавим колонки которых нет у нас
    df_predict["month"], df_predict["hour"] = time.month, time.hour
    df_predict["year"], df_predict["day"] = time.year, time.day
    df_predict["day_of_week"] = pd.DatetimeIndex(time).day_of_week
    df_predict = trans(df_predict)
    # Предикт вероятностей
    prediction = model.predict_proba(df_predict)[:, 1]

    # Добавление в датафрейм вероятностей
    df_predict['prediction'] = prediction

    # Уберем записи, где пользователь ранее ставил лайк
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = df_predict[~df_predict.index.isin(liked_posts)]

    # Сортировка вероятностей по убыванию
    df_predict = filtered_.sort_values(by='prediction', ascending=False).reset_index(drop=True)

    # Отбор постов
    df_predict = df_predict.loc[:limit - 1]

    return [PostGet(**{
        "id": int(i),
        "text": str(new_post[new_post.post_id == i].text.values[0]),
        "topic": str(new_post[new_post.post_id == i].topic.values[0])
    }) for i in df_predict.post_id]
