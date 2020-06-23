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
//const db = require('./db.js')

const app = express();

app.get('/', (req, res) => {
    res
    .status(200)
    .send('Hello Jumping, world!')
    .end();
});


app.get('/facebook/getPost', async (req, res) => {


    await scrapeGroup.getAllGroup();

    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/putComment', async (req, res) => {


    await scrapeComment.gotopage();

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
