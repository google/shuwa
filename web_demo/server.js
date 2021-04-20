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
