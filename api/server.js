// const express = require("express");
// const { exec } = require("child_process");
// const cors = require("cors");
// const multer = require("multer");
// const path = require("path");
// const fs = require("fs");
// const app = express();
// const port = 8080;

// app.use(cors());

// const uploadsFolder = path.resolve(__dirname, "../uploads");

// if (!fs.existsSync(uploadsFolder)) {
//   fs.mkdirSync(uploadsFolder, { recursive: true });
// }

// const storage = multer.diskStorage({
//   destination: uploadsFolder, // Save files in the root-level uploads folder
//   filename: (req, file, cb) => {
//     const ext = path.extname(file.originalname); // Get original file extension
//     const uniqueName = Date.now() + "-" + Math.round(Math.random() * 1e9) + ext; // Generate unique filename
//     cb(null, uniqueName);
//   },
// });

// const upload = multer({ storage });

// app.post("/get-tutorial", upload.single("image"), (req, res) => {
//   if (!req.file) {
//     return res.status(400).json({ error: "No file uploaded" });
//   }

//   const imagePath = path.join("uploads", req.file.filename);

//   // Call Python script with image path
//   exec(`python3 main.py "${imagePath}"`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`exec error: ${error}`);
//       return res.status(500).json({ error: "Failed to process image" });
//     }
//     if (stderr) {
//       console.error(`stderr: ${stderr}`);
//       return res.status(500).json({ error: "Error in Python script" });
//     }

//     try {
//       const tutorialData = JSON.parse(stdout); // Assuming Python script returns JSON
//       res.json(tutorialData);
//     } catch (e) {
//       res.status(500).json({ error: "Failed to parse tutorial data" });
//     }
//   });
// });

// app.listen(port, () => {
//   console.log(`Server running at http://localhost:${port}`);
// });

// // app.get("/get-tutorial", (req, res) => {
// //   const pythonProcess = spawn("python3", ["main.py", imagePath]);

// //   let data = "";
// //   pythonProcess.stdout.on("data", (chunk) => {
// //     data += chunk;
// //   });

// //   pythonProcess.stderr.on("data", (err) => {
// //     console.error(`Python Error: ${err}`);
// //   });

// //   pythonProcess.on("close", (code) => {
// //     if (code !== 0) {
// //       return res.status(500).json({ error: "Python script failed" });
// //     }
// //     try {
// //       res.json(JSON.parse(data));
// //     } catch (e) {
// //       res.status(500).json({ error: "Invalid JSON output from Python script" });
// //     }
// //   });
// // });

// // app.post("/get-tutorial", upload.single("image"), (req, res) => {

// //   console.log("req", req.file);
// //   if (!req.file) {
// //     return res.status(400).json({ error: "No file uploaded" });
// //   }

// //   const imagePath = path.join(uploadsFolder, req.file.filename); // Get full file path

// //   console.log("imagePath", imagePath);

// //   const command = `python3 main.py ${imagePath}`;
// //   console.log(`Executing command: ${command}`);

// //   exec(command, (error, stdout, stderr) => {
// //     if (error) {
// //       console.error(`exec error: ${error}`);
// //       return res.status(500).json({ error: "Failed to get tutorial" });
// //     }
// //     if (stderr) {
// //       console.error(`stderr: ${stderr}`);
// //       return res.status(500).json({ error: "Failed to get tutorial" });
// //     }

// //     console.log("Python script output:", stdout);

// //     try {
// //       const tutorialData = JSON.parse(stdout);
// //       console.log("tutorialData", tutorialData);
// //       res.json(tutorialData);
// //     } catch (e) {
// //       console.error("Error parsing tutorial data:", e);
// //       res.status(500).json({ error: "Failed to parse tutorial data" });
// //     }
// //   });
// // });

const express = require("express");
const { spawn } = require("child_process"); // Use spawn instead of exec
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const app = express();
const port = 8080;

app.use(cors());

const uploadsFolder = path.resolve(__dirname, "../uploads");

if (!fs.existsSync(uploadsFolder)) {
  fs.mkdirSync(uploadsFolder, { recursive: true });
}

const storage = multer.diskStorage({
  destination: uploadsFolder, // Save files in the root-level uploads folder
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname); // Get original file extension
    const uniqueName = Date.now() + "-" + Math.round(Math.random() * 1e9) + ext; // Generate unique filename
    cb(null, uniqueName);
  },
});

const upload = multer({ storage });

app.post("/get-tutorial", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const imagePath = path.join(__dirname, "../uploads", req.file.filename); // Full path to file

  console.log("Received file:", req.file.originalname); // Log original file name
  console.log("Processing image at:", imagePath); // Log full image path

  // Call Python script with the image path
  const pythonProcess = spawn("python3", ["main.py", imagePath]);

  let output = "";
  pythonProcess.stdout.on("data", (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error("Python error:", data.toString());
  });

  pythonProcess.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: "Python script failed" });
    }

    try {
      const tutorialData = JSON.parse(output); // Assuming Python script returns JSON
      res.json(tutorialData);
    } catch (e) {
      console.error("Error parsing tutorial data:", e);
      res.status(500).json({ error: "Failed to parse tutorial data" });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
