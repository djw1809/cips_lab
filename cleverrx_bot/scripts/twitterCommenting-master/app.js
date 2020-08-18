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


const getTwitterPost = require('./groups/getTwitter.js')
const tweetComment = require('./groups/commentTwitter.js');

const pageScrollLength = 1;
var randSampling = true; // if true selects random accounts from facebookAccounts.json.
var accountNo = 4; // selects specific account from facebookAccounts.json.
const app = express();


app.get('/', (req, res) => {
    res
    .status(200)
    .send('Come to decode facebook!')
    .end();
});


// URL: http://0.0.0.0:8082/facebook/tweetComment

app.get('/facebook/getTwitterPost', async (req, res) => {

    await getTwitterPost.gotopage(pageScrollLength);
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/tweetComment', async (req, res) => {

    await tweetComment.gotopage();
    res
        .status(200)
        .send('Twitter commenting done logged in!')
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
