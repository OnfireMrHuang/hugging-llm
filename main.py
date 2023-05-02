# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from embedding.clustering import Clustering


# if __name__ == '__main__':
#     X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
#     C, S = np.cos(X), np.sin(X)
#
#     plt.plot(X, C)
#     plt.plot(X, S)
#
#     plt.show()


if __name__ == '__main__':
    clustering = Clustering()
    clustering.classification()
