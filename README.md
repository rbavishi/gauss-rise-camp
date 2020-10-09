## Gauss (AutoPandas) - Rise Camp 2020

### Instructions for Docker

Download the image using the following command
```
docker pull rbavishi/risecamp2020gauss:dr1
```

Run the jupyter notebook server using the following command
```
docker run -p 8889:8888 rbavishi/risecamp2020gauss:dr1
```

Now visit `localhost:8889` in a browser to access the server. Note that if you wish to use a different port, replace `8889` in the command above with the port of your choice. **The password/token for this container is `rc2020gauss`**

### Instructions for Binder

Click on the link below to launch a binder instance.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rbavishi/gauss-rise-camp/master?filepath=Gauss-AutoPandas-Tutorial.ipynb)
