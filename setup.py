from setuptools import setup

setup(name='ncdbad-predictor',
      version="0.1",
      author='Benedetto Marco Serinelli',
      author_email='Benedetto.Serinelli@unige.ch',
      url='https://isec.unige.ch/',
      platforms=['any'],
      install_requires=['joblib', 'keras', 'scikit-learn==0.23.1', 'xgboost', 'tensorflow', 'watchdog', 'pandas',
                        'protobuf==3.12.2', 'pyzmq==19.0.1', 'numpy', 'python-socketio[client]', 'six==1.15.0', 'numpy=1.19.2','h5py=2.10.0'])
