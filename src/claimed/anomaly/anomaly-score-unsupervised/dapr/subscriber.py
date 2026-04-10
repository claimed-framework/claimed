#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install dapr==1.7.0 cloudevents==1.6.1 dapr-ext-grpc==1.7.0')


# In[2]:


#import sys
#s#ys.path.append('/home/romeokienzler/venvs/claimed/lib/python3.10/site-packages/')
#sys.path


# In[ ]:


#dependencies
from cloudevents.sdk.event import v1
from dapr.ext.grpc import App
import logging
import json

#code
app = App()
logging.basicConfig(level = logging.INFO)
#Subscribe to a topic 
@app.subscribe(pubsub_name='pubsub', topic='anomaly-data')
def mytopic(event: v1.Event) -> None:
    data = json.loads(event.Data())
    logging.info('Subscriber received: ' + str(data))

app.run(6002)


# In[ ]:




