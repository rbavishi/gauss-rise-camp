## Gauss (AutoPandas) - Rise Camp 2020

We provide both docker and in-browser options below. The in-browser option is powered by the excellent infrastructure provided by MyBinder (https://mybinder.org).

**IMPORTANT** : If you have docker installed and are able to download images and run containers using the shell commands below, please use the docker option. Binder only allows 100 users concurrently and hence is only intended to be a backup for people who do not have docker installed or on a system such as Windows with limited Docker support.

### Instructions for Docker

Download the image using the command below. Although the image is small (~400MB) it would be best to download this *before* the tutorial session.
```
docker pull rbavishi/risecamp2020gauss:latest
```

Run the jupyter notebook server using the following command
```
docker run -p 8889:8888 rbavishi/risecamp2020gauss:latest
```

Now visit `localhost:8889` in a browser to access the server. Note that if you wish to use a different port, replace `8889` in the command above with the port of your choice. **The password/token for this container is `rc2020gauss`**

### Instructions for Binder

Click on the link below to launch a binder instance.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rbavishi/gauss-rise-camp/master?filepath=Gauss-AutoPandas-Tutorial.ipynb)

If you are unable to launch an instance due to instance limits, try *one* of the backup links below - 

Backup Link 1 - [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rbavishi/gauss-rise-camp-1/master?filepath=Gauss-AutoPandas-Tutorial.ipynb)

Backup Link 2 - [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rbavishi/gauss-rise-camp-2/master?filepath=Gauss-AutoPandas-Tutorial.ipynb)

**NOTE** - The backup links (and the corresponding repos) above will be removed after the tutorial (10/29/2020 12:45 PM - 2:00 PM PST) and are only intended to handle any unexpected surge.
