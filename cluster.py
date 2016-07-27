import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from pymongo import MongoClient
import calendar

client = MongoClient('localhost', 27017)
db = client['promed']
posts = db.posts

def get_articles(disease):
  print('get articles', disease)
  articles = posts.find({'zoomLat': {'$ne': None}, 'subject.diseaseLabels':{'$not':{'$size': 0}}, 'subject.diseaseLabels': {'$in': [disease]}},{'subject.diseaseLabels':1,'zoomLat': 1, 'zoomLon': 1, 'sourceDate': 1, 'promedDate': 1}).limit(100)
  articles = list(articles);
  for article in articles:
    try:
      # not all articles have a sourceDate so fall back to promedDate if missing.
      date = article['sourceDate'] or article['promedDate']
      # convert date object to timestamp so DBSCAN can handle it
      article['sourceDate'] = calendar.timegm(date.timetuple())/100000.0
      # convert disease labels array to single disease name
      article['subject'] = article['subject']['diseaseLabels'][0]
    except Exception as e:
      print("Problem parsing article:", article)
      print(e)
      raise
  print(sorted([x['sourceDate'] for x in articles]))

  return articles

def get_disease_list():
  diseaseNames = posts.distinct('subject.diseaseLabels')
  return sorted(diseaseNames)
  # print(sorted(diseaseNames))
  # print(sorted(set(articles['subject'])))

def cluster_data(df):

  def plot_results():
    unique_labels = set(db.labels_)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
      if k == -1:
          # Black used for noise.
          col = 'k'

      class_member_mask = (unique_labels == k)

      plt.plot(df['zoomLon'], df['zoomLat'], 'o', markerfacecolor=col,
               markeredgecolor='k', markersize=14)

    plt.title('Estimated number of clusters: %d' % num_clusters)
    plt.show()

  # coordinates = df.as_matrix(columns=['zoomLon', 'zoomLat', 'sourceDate'])
  coordinates = df.as_matrix(columns=['zoomLon', 'zoomLat'])
  db = DBSCAN(eps=.1, min_samples=1, algorithm='ball_tree').fit(coordinates)
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  cluster_labels = db.labels_
  num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
  clusters = pd.Series([coordinates[cluster_labels == n] for n in range(num_clusters)])
  plot_results()
  print('Number of clusters: {}'.format(num_clusters))
  print('Cluster names', cluster_labels)

if __name__ == '__main__':
  diseaseList = get_disease_list()
  for disease in diseaseList:
    articleList = list(get_articles(disease))
    print("{0} articles for {1}".format(len(articleList), disease))
    df = pd.DataFrame(articleList)
    cluster_data(df)