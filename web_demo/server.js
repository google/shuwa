/**
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

var express = require("express");
var http = require("http");
var app = express();
var path = require("path");

app.use(express.static(__dirname + "/public"));

app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname + "/index.html"));
});

const port = 8000;

const server = http.createServer(app);
server.listen(port, () => {
  console.log(
    `Example app listening on port ${port}! Go to http://localhost:${port}/`
  );
});
