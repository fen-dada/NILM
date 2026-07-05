import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectDir = "E:/NILM_Project";
const eventsPath = path.join(projectDir, "outputs/events/candidate_events.csv");
const outputDir = path.join(projectDir, "outputs/labels");
const outputPath = path.join(outputDir, "event_labels.xlsx");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes && ch === '"' && next === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && ch === ",") {
      row.push(cell);
      cell = "";
    } else if (!inQuotes && (ch === "\n" || ch === "\r")) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function safeImageName(index, row) {
  const rel = row.relative_path.replace(/[<>:"/\\|?*\s]+/g, "_");
  const category = row.category.replace(/[<>:"/\\|?*\s]+/g, "_");
  return `${String(index).padStart(3, "0")}_${category}_${rel}.png`;
}

function toObjects(rows) {
  const headers = rows[0].map((header) => header.replace(/^\uFEFF/, ""));
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

const csvText = await fs.readFile(eventsPath, "utf8");
const events = toObjects(parseCsv(csvText)).slice(0, 20);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("事件标注");
const guide = workbook.worksheets.add("填写说明");

const headers = [
  "编号",
  "区域",
  "事件时间",
  "判断",
  "可能设备",
  "备注",
  "事件分数",
  "触发特征",
  "事件前文件",
  "事件后文件",
  "复核图片",
];

const data = events.map((event, index) => {
  const eventNo = index + 1;
  const imagePath = path.join(projectDir, "outputs/event_review", safeImageName(eventNo, event)).replaceAll("\\", "/");
  return [
    eventNo,
    event.category,
    event.timestamp,
    "",
    "",
    "",
    Number(event.event_score),
    event.triggered_features,
    event.previous_relative_path,
    event.relative_path,
    imagePath,
  ];
});

sheet.getRange("A1:K1").values = [headers];
sheet.getRangeByIndexes(1, 0, data.length, headers.length).values = data;

sheet.getRange("A1:K1").format = {
  fill: "#1F4E79",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
sheet.getRange(`A1:K${data.length + 1}`).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2EC",
};
sheet.getRange(`A2:K${data.length + 1}`).format = {
  fill: "#FFFFFF",
  font: { color: "#1F2933" },
  wrapText: true,
};
sheet.getRange(`D2:F${data.length + 1}`).format.fill = "#FFF7CC";
sheet.getRange(`G2:G${data.length + 1}`).format.numberFormat = "0.00";
sheet.getRange(`A2:A${data.length + 1}`).format.numberFormat = "0";
sheet.getRange("A:A").format.columnWidth = 8;
sheet.getRange("B:B").format.columnWidth = 14;
sheet.getRange("C:C").format.columnWidth = 24;
sheet.getRange("D:D").format.columnWidth = 14;
sheet.getRange("E:E").format.columnWidth = 20;
sheet.getRange("F:F").format.columnWidth = 32;
sheet.getRange("G:G").format.columnWidth = 12;
sheet.getRange("H:H").format.columnWidth = 34;
sheet.getRange("I:J").format.columnWidth = 38;
sheet.getRange("K:K").format.columnWidth = 72;
sheet.getRange(`A2:K${data.length + 1}`).format.rowHeight = 42;
sheet.freezePanes.freezeRows(1);

sheet.getRange(`D2:D${data.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["启动", "停止", "状态切换", "不确定", "非事件"] },
};

sheet.tables.add(`A1:K${data.length + 1}`, true, "EventLabels");

guide.getRange("A1:C1").values = [["字段", "怎么填", "例子"]];
guide.getRange("A2:C6").values = [
  ["判断", "从下拉框选择事件类型。先粗略标也可以。", "启动"],
  ["可能设备", "如果知道现场设备，就写设备名；不知道就写不确定。", "老化房负荷 / 不确定"],
  ["备注", "写你看图得到的依据或现场信息。", "三相电流从接近 0 变大"],
  ["事件分数", "程序给出的变化强度，只用于排序，不需要修改。", "5651.49"],
  ["复核图片", "本地图片路径，打开后可看事件前后波形。", "E:/NILM_Project/outputs/event_review/...png"],
];
guide.getRange("A1:C1").format = {
  fill: "#1F4E79",
  font: { bold: true, color: "#FFFFFF" },
};
guide.getRange("A1:C6").format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2EC",
};
guide.getRange("A:C").format.wrapText = true;
guide.getRange("A:A").format.columnWidth = 16;
guide.getRange("B:B").format.columnWidth = 46;
guide.getRange("C:C").format.columnWidth = 40;
guide.freezePanes.freezeRows(1);

const inspect = await workbook.inspect({
  kind: "table",
  range: "事件标注!A1:K6",
  include: "values",
  tableMaxRows: 6,
  tableMaxCols: 11,
  maxChars: 4000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "事件标注", range: "A1:K10", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "event_labels_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(`saved ${outputPath}`);
