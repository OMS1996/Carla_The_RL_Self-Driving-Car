# Carla the Reinforcement Self-Driving-Car (Version Morra)
<p>In this project I aim to create a self-driving car that uses a reinforcement learning approach to navigate in an open-source simulator for autonomous driving research called Carla. The data that the car will use as to guide its decision is image data in the `RGB` format and `collision` data. </p>
For more information about carla please visit the following Links:
- https://carla.org/
- https://github.com/carla-simulator/carla
![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/carla_desktop0.PNG?raw=true)

# Motivation
<p> Created this project as part of my Master's Final project for the Year 2020 and a passion for reinforcement learning. </p>
This video from openAI really inspired me: https://www.youtube.com/watch?v=kopoLzvh5jY

### How to use this repo
- Download anaconda
- Create a virtual environment: conda create -n envname python=3.7 anaconda
- pip install requirements.txt
- Download the Carla Repo from https://github.com/carla-simulator/carla
- Once everything is setup you must ensure that you have that you have CarlaUE4.exe running. or if you are on linux run the command ./CarlaUE4.sh

# What is in this repo
- Code for a reinforcement learning self-driving car.
- A step by step code breakdown in the form of a jupyter notebook.
- A modularized version of the code.
- Powerpoint presentation.
- Data and Graphs.
- A Readme with detailed instructions.

### Carla

![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/carla_look1.PNG?raw=true)

This is how carla looks like from the inside. It is an extremely beautiful environment.

### Reinforcement Learning 
The main idea in RL is that you have an agent which is an "intelligent being" the interacts with an environment by means of taking actions and then receives feedback from the environment to indicate whether the agent has done well or bad. Like raising a  child , if he does well in school you encourage(REWARD) him if he doesn’t then you perhaps ground him (Penalize). and your child starts to adjust his behavior accordingly.

Note that a +ve reward indicates a reward and a -ve reward indicates penalty.

![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/rl_env1.PNG)

# DQN
How the DQN algorithm generally looks like is as follows: courtesy of @deeplizard's website: https://deeplizard.com/learn/video/0bt0SjbS3xc
<!-- BLOG-POST-LIST:START -->
<!-- BLOG-POST-LIST:END -->
```
1.Initialize replay memory capacity.
2.Initialize the network with random weights.
3.For each episode:
  1.Initialize the starting state.
  2.For each time step:
    1.Select an action.
      Via exploration or exploitation
    2.Execute selected action in an emulator.
    3.Observe reward and next state.
    4.Store experience in replay memory.
    5.Sample random batch from replay memory.
    6.Preprocess states from batch.
    7.Pass batch of preprocessed states to policy network.
    8.Calculate loss between output Q-values and target Q-values.
      Requires a second pass to the network for the next state
    9.Gradient descent updates weights in the policy network to minimize loss."
```


### Demo video ( First few episodes )
Here is the first few minutes of the training process for self driving car.
Please see the [video](https://www.youtube.com/watch?v=oAbDeb887_U) by [@OMS1996](https://github.com/OMS1996).
As you can see at the beginning it is not very smart but slowly but surely it begins to get smarter and smarter.

<details>
  <summary>Results!</summary>

  ![advanced](https://.png)
</details>

#### Potential improvements.
- [ ] Incoporate dynamic weather for a wider range of data. ( Level: Easy)
- [ ] Implement prioritized experience replay ( Level: Medium)
- [ ] Create a perception system (Level: Hard)
- [ ] Attempt an improved DDQN
- [ ] Implement PPO 
- [ ] Implement A3C
- [ ] Create a model based self-driving car (Level: Hard)
- [ ] Combine RL + Rule based machine learning for self-driving car (level: Very hard)
- [ ] Use imitation learning

### Bugs
If you are experiencing any bugs, please email me at omarmoh.said@yahoo.com

### Resources
Following are the list of some popular blogging platforms and their RSS feed urls:

| Name | Feed URL | Comments 
|--------|--------|--------
| [Dev.to](https://dev.to/) | `https://dev.to/feed/username` | Replace username with your own username 
| [Wordpress](https://wordpress.org/) | `https://www.example.com/feed/` | Replace with your own blog url 
| [Medium](https://medium.com/) | `https://medium.com/feed/@username` | Replace @username with your Medium username 
| [Stackoverflow](https://stackoverflow.com/) | `https://stackoverflow.com/feeds/user/userid` (https://meta.stackexchange.com/questions/98771/what-is-my-user-id/111130#111130) | https://stackoverflow.com/feeds/user/5283532 |
| [StackExchange](https://stackexchange.com/) | `https://subdomain.stackexchange.com/feeds/user/userid` (https://meta.stackexchange.com/questions/98771/what-is-my-user-id/111130#111130) and sub-domain | https://devops.stackexchange.com/feeds/user/15 |
| [Ghost](https://ghost.org/) | `https://www.example.com/rss/` | Replace with your own blog url 
| [Drupal](https://www.drupal.org/) | `https://www.example.com/rss.xml` | Replace with your own blog url 
| [Youtube Playlists](https://www.youtube.com) | `https://www.youtube.com/feeds/videos.xml?playlist_id=playlistId` | Replace `playlistId` with your own Youtube playlist id | https://www.youtube.com/feeds/videos.xml?playlist_id=PLJNqgDLpd5E69Kc664st4j7727sbzyx0X |
| [Youtube Channel Video list](https://www.youtube.com) |  `https://www.youtube.com/feeds/videos.xml?channel_id=channelId` | Replace `channelId` with your own Youtube channel id | https://www.youtube.com/feeds/videos.xml?channel_id=UCDCHcqyeQgJ-jVSd6VJkbCw |
| [Anchor.fm Podcasts](https://anchor.fm/) | `https://anchor.fm/s/podcastId/podcast/rss` | You can get the rss feed url of a podcast by following [these](https://help.anchor.fm/hc/en-us/articles/360027712351-Locating-your-Anchor-RSS-feed) instructions 
| [Hashnode](https://hashnode.com/) | `https://@username.hashnode.dev/rss.xml` | Replace @username with your Hashnode username | https://polilluminato.hashnode.dev/rss.xml |
| [Google Podcasts](https://podcasts.google.com/) | `https://podcasts.google.com/feed/channelId` | Replace `channelId` with your Google podcast channel Id | https://podcasts.google.com/feed/aHR0cHM6Ly9mZWVkcy5zb3VuZGNsb3VkLmNvbS91c2Vycy9zb3VuZGNsb3VkOnVzZXJzOjYyOTIxMTkwL3NvdW5kcy5yc3M= |
| [Reddit](https://www.reddit.com/) | `http://www.reddit.com/r/topic/.rss` | You can create an RSS feed by adding '.rss' to the end of an existing Reddit URL. Replace `topic` with SubReddit topic that interest you or localized to you.| http://www.reddit.com/r/news/.rss |
| [Analytics India Magazine](https://analyticsindiamag.com/) | `https://analyticsindiamag.com/author/author_name/feed/` | Replace `author_name` with your name | https://analyticsindiamag.com/author/kaustubhgupta1828gmail-com/feed/ |
| [Feedburner](https://feedburner.com/) | `https://feeds.feedburner.com/feed_address` | Replace `feed_address` with your Feedburner feed address 

### Thanks 
- Director of my program Professor.Andy Catlin
- Dean Paul Russo
- My supervisor Dr. Wonjun


### My Github account
* [My own GitHub profile readme](https://github.com/OMS1996) 

### Liked it?
Hope you liked this project, don't forget to give it a star ⭐



