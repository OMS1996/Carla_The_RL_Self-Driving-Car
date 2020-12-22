# Carla the Reinforcement Self-Driving-Car (Version Morra)
In this project I aim to create a self-driving car that uses a reinforcement learning approach to navigate in an open-source simulator for autonomous driving research called Carla. The data that the car will use as to guide its decision is image data in the `RGB` format and `collision` data. 

For more information about carla please visit the following Links:
- https://carla.org/
- https://github.com/carla-simulator/carla
![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/carla_desktop0.PNG?raw=true)

# Motivation
Created this project as part of my Master's thesis for the Year 2020

### How to use this repo
- Download anaconda
- Create a virtual environment: conda create -n envname python=3.7 anaconda
- pip install requirements.txt
- Download the Carla Repo from https://github.com/carla-simulator/carla
- Once everything is setup you must ensure that you have that you have CarlaUE4.exe running. or if you are on linux run the command ./CarlaUE4.sh


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
- Replace the above url list with your own rss feed urls. See [popular-sources](#popular-sources) for a list of common RSS feed urls.
- Commit and wait for it to run automatically or you can also trigger it manually to see the result instantly. To trigger the workflow manually, please follow the steps in the [video](https://www.youtube.com/watch?v=ECuqb5Tv9qI&t=272s).


### Reinforcement Learning 
![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/rl_env1.PNG)

### Carla
![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/carla_look1.PNG?raw=true)

# StackOverflow Activity
<!-- STACKOVERFLOW:START -->
<!-- STACKOVERFLOW:END -->
```
- Create `stack-overflow-workflow.yml` in your `workflows` folder with the following contents, replace **4214976** with your StackOverflow [user id](https://meta.stackexchange.com/questions/98771/what-is-my-user-id/111130#111130):
```yaml
name: Latest stack overflow activity
on:
  schedule:
    # Runs every 5 minutes
    - cron: '*/5 * * * *'

jobs:
  update-readme-with-stack-overflow:
    name: Update this repo's README with latest activity from StackOverflow
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: gautamkrishnar/blog-post-workflow@master
        with:
          comment_tag_name: "STACKOVERFLOW"
          commit_message: "Updated readme with the latest stackOverflow data"
          feed_list: "https://stackoverflow.com/feeds/user/4214976"
```
<details>
  <summary>See the result!</summary>

  ![advanced](https://user-images.githubusercontent.com/8397274/88197889-b727ff80-cc60-11ea-8e4a-b1fbd8dd9d06.png)
</details>

### Examples 
* [My own GitHub profile readme](https://github.com/OMS1996) 

### Demo video
Here is the first few minutes of the training process for self driving car.
Please see the [video](https://www.youtube.com/watch?v=oAbDeb887_U) by [@OMS1996](https://github.com/OMS1996).

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

| Name | Feed URL | Comments | Example |
|--------|--------|--------|--------|
| [Dev.to](https://dev.to/) | `https://dev.to/feed/username` | Replace username with your own username | https://dev.to/feed/gautamkrishnar |
| [Wordpress](https://wordpress.org/) | `https://www.example.com/feed/` | Replace with your own blog url | https://www.gautamkrishnar.com/feed/ |
| [Medium](https://medium.com/) | `https://medium.com/feed/@username` | Replace @username with your Medium username | https://medium.com/feed/@khaosdoctor |
| [Stackoverflow](https://stackoverflow.com/) | `https://stackoverflow.com/feeds/user/userid` | Replace with your StackOverflow [UserId](https://meta.stackexchange.com/questions/98771/what-is-my-user-id/111130#111130) | https://stackoverflow.com/feeds/user/5283532 |
| [StackExchange](https://stackexchange.com/) | `https://subdomain.stackexchange.com/feeds/user/userid` | Replace with your StackExchange [UserId](https://meta.stackexchange.com/questions/98771/what-is-my-user-id/111130#111130) and sub-domain | https://devops.stackexchange.com/feeds/user/15 |
| [Ghost](https://ghost.org/) | `https://www.example.com/rss/` | Replace with your own blog url | https://blog.codinghorror.com/rss/ |
| [Drupal](https://www.drupal.org/) | `https://www.example.com/rss.xml` | Replace with your own blog url | https://www.arsenal.com/rss.xml |
| [Youtube Playlists](https://www.youtube.com) | `https://www.youtube.com/feeds/videos.xml?playlist_id=playlistId` | Replace `playlistId` with your own Youtube playlist id | https://www.youtube.com/feeds/videos.xml?playlist_id=PLJNqgDLpd5E69Kc664st4j7727sbzyx0X |
| [Youtube Channel Video list](https://www.youtube.com) |  `https://www.youtube.com/feeds/videos.xml?channel_id=channelId` | Replace `channelId` with your own Youtube channel id | https://www.youtube.com/feeds/videos.xml?channel_id=UCDCHcqyeQgJ-jVSd6VJkbCw |
| [Anchor.fm Podcasts](https://anchor.fm/) | `https://anchor.fm/s/podcastId/podcast/rss` | You can get the rss feed url of a podcast by following [these](https://help.anchor.fm/hc/en-us/articles/360027712351-Locating-your-Anchor-RSS-feed) instructions | https://anchor.fm/s/1e784a38/podcast/rss |
| [Hashnode](https://hashnode.com/) | `https://@username.hashnode.dev/rss.xml` | Replace @username with your Hashnode username | https://polilluminato.hashnode.dev/rss.xml |
| [Google Podcasts](https://podcasts.google.com/) | `https://podcasts.google.com/feed/channelId` | Replace `channelId` with your Google podcast channel Id | https://podcasts.google.com/feed/aHR0cHM6Ly9mZWVkcy5zb3VuZGNsb3VkLmNvbS91c2Vycy9zb3VuZGNsb3VkOnVzZXJzOjYyOTIxMTkwL3NvdW5kcy5yc3M= |
| [Reddit](https://www.reddit.com/) | `http://www.reddit.com/r/topic/.rss` | You can create an RSS feed by adding '.rss' to the end of an existing Reddit URL. Replace `topic` with SubReddit topic that interest you or localized to you.| http://www.reddit.com/r/news/.rss |
| [Analytics India Magazine](https://analyticsindiamag.com/) | `https://analyticsindiamag.com/author/author_name/feed/` | Replace `author_name` with your name | https://analyticsindiamag.com/author/kaustubhgupta1828gmail-com/feed/ |
| [Feedburner](https://feedburner.com/) | `https://feeds.feedburner.com/feed_address` | Replace `feed_address` with your Feedburner feed address | https://feeds.feedburner.com/darkwood-fr/blog |

### Thanks 
- Director of my program Professor.Andy Catlin
- Dean Paul Russo
- My supervisor Dr. Wonjun

### Liked it?
Hope you liked this project, don't forget to give it a star ⭐



