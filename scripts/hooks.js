import fs from "fs";
import path from "path";

const logDir = path.resolve("./reports/logs");
const logFile = path.join(logDir, "error_log.txt");

// Ensure log directory exists
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });

export function afterResponse(req, res, context, ee, next) {
  if (res.statusCode >= 400) {
    const entry = `[${new Date().toISOString()}] ${req.method} ${req.url} -> ${res.statusCode}\nResponse: ${JSON.stringify(res.body)}\n\n`;
    fs.appendFileSync(logFile, entry);
  }
  return next();
}
