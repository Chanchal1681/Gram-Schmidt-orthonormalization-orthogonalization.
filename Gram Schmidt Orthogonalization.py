#!/usr/bin/env python
# coding: utf-8

# ##  1.  Let x1_=(1, 0,1,1)' , X 2_= ( -1, 0, 1,1)' , X3_=(0,-1,1,1)'  and  X4_=(0, 0,1, 0)'  are  linearly independent  set of  vectors of  R^4 . Find  an orthonormal  basis of 4  starting with X 4 , X 3 , X 2 , X 1.

# In[37]:


import numpy as np
x4=[0, 0, 1, 0]
x3=[0, -1, 1, 1]
x2=[-1, 0, -1, 1]
x1=[1, 0, 1, 1]


# #### Finding norm of each vector 

# In[38]:


z1=x1/np.linalg.norm(x1)
z2=x2/np.linalg.norm(x2)
z3=x3/np.linalg.norm(x3)
z4=x4/np.linalg.norm(x4)
np.array([z1,z2,z3,z4])


# #### Gram Schmidt Orthogonalization

# In[39]:


y1=z4
y2=x3-np.inner(x4,y1)/np.inner(y1,y1)*z4
np.round(np.inner(y1,y2),3)


# In[40]:


y3=x2-np.inner(x2,y2)/np.inner(y2,y2)*y2-np.inner(x2,y1)/np.inner(y1,y1)*y1
np.round(np.inner(y2,y3),3)


# In[43]:


y4=x1-np.inner(x1,y1)/np.inner(y1,y1)*y1-np.inner(x1,y2)/np.inner(y2,y2)*y2-np.inner(x1,y3)/np.inner(y3,y3)*y3
np.round(np.inner(y3,y4),3)


# #### Orthonormal basis starting with x4,x3,x2,x1 are 

# In[45]:


np.array([y1,y2,y3,y4])


# ## 2. Obtain an 4 by 4 orthogonal matrix with all elements in first row equal
# 
# 

# In[48]:


np.random.randint(0,2,(6,6)) # randomized orthogonal matrix


# In[49]:


x4=[1, 1, 1, 1]
x3=[0, -1, 1, 1]
x2=[-1, 0, -1, 1]
x1=[1, 0, 1, 1]
z1=x1/np.linalg.norm(x1)
z2=x2/np.linalg.norm(x2)
z3=x3/np.linalg.norm(x3)
z4=x4/np.linalg.norm(x4)
np.array([z1,z2,z3,z4])


# In[50]:


y1=z4
y2=x3-np.inner(x4,y1)/np.inner(y1,y1)*z4
y3=x2-np.inner(x2,y2)/np.inner(y2,y2)*y2-np.inner(x2,y1)/np.inner(y1,y1)*y1
y4=x1-np.inner(x1,y1)/np.inner(y1,y1)*y1-np.inner(x1,y2)/np.inner(y2,y2)*y2-np.inner(x1,y3)/np.inner(y3,y3)*y3
np.array([y1,y2,y3,y4])


# ## 3.Using Gram-Schmidt orthogonalization process, construct an ortho-normal basis from the following basis {(2,3,0),(6,1,0),(0,2,4)}.
# 
# 

# In[51]:


x1=[[2,-1,1]]
x2=[[4,2,1]]
x3=[[-1,3,2]]


# #### For finding norm each of vector

# In[53]:


z1=x1/np.linalg.norm(x1)
z2=x2/np.linalg.norm(x2)
z3=x3/np.linalg.norm(x3)
np.array([z1,z2,z3])


# #### For GSO

# In[55]:


y1=z1
y2=x2-np.inner(x2,y1)/np.inner(y1,y1)*y1
y3=x3-np.inner(x3,y2)/np.inner(y2,y2)*y2-np.inner(x3,y1)/np.inner(y1,y1)*y1
#verification
np.round(np.inner(y1,y2),3)
np.round(np.inner(y2,y3),3)
np.round(np.inner(y1,y3),3)


# #### Orthonormal basis are

# In[56]:


np.array([y1,y2,y3])


# In[ ]:




