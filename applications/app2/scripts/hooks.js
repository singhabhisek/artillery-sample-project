const fs = require("fs");
const path = require("path");

const LOG_DIR = path.resolve("./artillery-results/logs");
if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });

const ERROR_LOG_FILE = path.join(LOG_DIR, "error_log.txt");

function logIfError(requestParams, response, context, ee, next) {
  try {
    const status = response.statusCode || 0;
    if (status !== 200) {
      const entry = {
        timestamp: new Date().toISOString(),
        url: requestParams.url,
        method: requestParams.method,
        status,
        responseBody: response.body ? response.body.toString() : "<empty>",
      };
      fs.appendFileSync(ERROR_LOG_FILE, JSON.stringify(entry) + "\n");
      console.error(`❌ Error logged: ${requestParams.url} → ${status}`);
    }
  } catch (err) {
    console.error("Hook error:", err);
  }
  return next();
}

module.exports = { logIfError };
