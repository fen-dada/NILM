import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectDir = "E:/NILM_Project";
const inputPath = path.join(projectDir, "outputs/labels/auto_event_labels.csv");
const outputDir = path.join(projectDir, "outputs/labels");
const outputPath = path.join(outputDir, "auto_event_labels.xlsx");

const csvText = (await fs.readFile(inputPath, "utf8")).replace(/^\uFEFF/, "");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "自动伪标注" });
const sheet = workbook.worksheets.getItem("自动伪标注");

const used = sheet.getUsedRange(true);
const values = used.values;
const rowCount = values.length;
const colCount = values[0]?.length ?? 0;

sheet.getRangeByIndexes(0, 0, 1, colCount).format = {
  fill: "#1F4E79",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
sheet.getRangeByIndexes(0, 0, rowCount, colCount).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2EC",
};
sheet.getRangeByIndexes(1, 0, Math.max(rowCount - 1, 1), colCount).format = {
  fill: "#FFFFFF",
  font: { color: "#1F2933" },
  wrapText: true,
};

sheet.getRange("A:A").format.columnWidth = 8;
sheet.getRange("B:B").format.columnWidth = 14;
sheet.getRange("C:C").format.columnWidth = 24;
sheet.getRange("D:D").format.columnWidth = 14;
sheet.getRange("E:E").format.columnWidth = 10;
sheet.getRange("F:G").format.columnWidth = 14;
sheet.getRange("H:H").format.columnWidth = 34;
sheet.getRange("I:I").format.columnWidth = 18;
sheet.getRange("J:J").format.columnWidth = 44;
sheet.getRange("K:K").format.columnWidth = 22;
sheet.getRange("L:S").format.columnWidth = 20;
sheet.getRange("T:U").format.columnWidth = 38;
sheet.getRange("V:V").format.columnWidth = 72;
sheet.getRangeByIndexes(1, 0, Math.max(rowCount - 1, 1), colCount).format.rowHeight = 36;
sheet.freezePanes.freezeRows(1);

if (rowCount > 1 && colCount > 0) {
  sheet.tables.add(`A1:V${rowCount}`, true, "AutoEventLabels");
}

const guide = workbook.worksheets.add("说明");
guide.getRange("A1:B1").values = [["项目", "说明"]];
guide.getRange("A2:B6").values = [
  ["标签性质", "自动伪标注：只基于电压电流特征，不是真实启停记录。"],
  ["启动", "平均三相电流由低到高，前后差异很大。"],
  ["停止", "平均三相电流由高到低，前后差异很大。"],
  ["状态切换", "事件前后都存在负荷，但电流或功率特征明显变化。"],
  ["不确定", "变化方向或幅度不足以稳定判断。"],
];
guide.getRange("A1:B1").format = {
  fill: "#1F4E79",
  font: { bold: true, color: "#FFFFFF" },
};
guide.getRange("A1:B6").format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2EC",
};
guide.getRange("A:A").format.columnWidth = 16;
guide.getRange("B:B").format.columnWidth = 68;
guide.getRange("A:B").format.wrapText = true;

const inspect = await workbook.inspect({
  kind: "table",
  range: "自动伪标注!A1:V6",
  include: "values",
  tableMaxRows: 6,
  tableMaxCols: 22,
  maxChars: 5000,
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
const preview = await workbook.render({ sheetName: "自动伪标注", range: "A1:V10", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "auto_event_labels_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(`saved ${outputPath}`);
