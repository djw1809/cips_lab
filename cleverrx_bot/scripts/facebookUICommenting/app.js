// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

// [START gae_node_request_example]
const express = require('express');
const scrapeGroup = require('./groups/getPosts.js')
const scrapeComment = require('./groups/postComment.js')
const getYesPosts = require('./groups/getYesterdayPosts.js')
const singleAccount = require('./groups/singleAccount.js')

const getPublicPost = require('./groups/getPublicPosts.js')



const pageScrollLength = 1;
var randSampling = true; // if true selects random accounts from facebookAccounts.json.
var accountNo = 1; // selects specific account from facebookAccounts.json.
const app = express();


app.get('/', (req, res) => {
    res
    .status(200)
    .send('Come to decode facebook!')
    .end();
});


// URL: http://0.0.0.0:8082/facebook/getLatestPost

app.get('/facebook/getLatestPost', async (req, res) => {

    await scrapeGroup.getAllGroup(pageScrollLength);
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/publicPosts', async (req, res) => {

    await getPublicPost.getAllGroup(pageScrollLength);
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/randomAccount', async (req, res) => {

    await scrapeComment.gotopage(randSampling);

    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/singleAccount', async (req, res) => {

    await singleAccount.gotopage(accountNo);
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});


app.get('/facebook/getYesterdayPost', async (req, res) => {
    
    await getYesPosts.getAllGroup(pageScrollLength)
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});




const http = require('http');

const hostname = '0.0.0.0';


// Start the server
const PORT = process.env.PORT || 8082;
app.listen(PORT,hostname, () => {
  console.log(`App listening on port  ${hostname} ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});
// [END gae_node_request_example]

module.exports = app;
