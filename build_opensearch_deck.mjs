import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const SKILL_DIR = 'C:/Users/ACER/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const workspaceDir = 'C:/Users/ACER/Code/TMG301';
const TMP_DIR = path.join(workspaceDir, '.opensearch-build');
const OUT_DIR = path.join(workspaceDir, 'outputs');
const FINAL_PPTX = path.join(OUT_DIR, 'OpenSearch_Operating_Principles_Team3.pptx');
const generated = 'C:/Users/ACER/.codex/generated_images/01a08536-0605-7a50-b5d7-84ee9d05271a/exec-9be59a49-7c89-4803-b968-927227a5898b.png';

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(OUT_DIR, { recursive: true });
const { resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({ fontFamily: 'Aptos' });
const ppt = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const W = 1280, H = 720;
const C = { navy:'#07162F', navy2:'#0D2447', cyan:'#48D8FF', blue:'#2E7CF6', amber:'#FFB938', white:'#F4F8FF', muted:'#AFC2DC', line:'#29466D', green:'#6FE0B0', red:'#FF7D85' };
const imgBytes = new Uint8Array(await fs.readFile(generated));

function rect(slide, x,y,w,h, fill, radius=16, line='none') {
  return slide.shapes.add({ geometry: radius ? 'roundRect' : 'rect', position:{left:x,top:y,width:w,height:h}, fill, line: line==='none'?{fill:'none',width:0}:{fill:line,width:1} });
}
function text(slide, s, x,y,w,h, size=24, color=C.white, opts={}) {
  const sh = slide.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h}, fill:'none', line:{fill:'none',width:0} });
  sh.text = s;
  sh.text.style = { typeface:font, fontSize:size, color, bold:!!opts.bold, italic:!!opts.italic, autoFit:'shrink', verticalAlignment:opts.valign||'middle', margin: opts.margin ?? 0 };
  return sh;
}
function title(slide, kicker, heading, sub='') {
  text(slide, kicker.toUpperCase(), 64, 34, 1100, 24, 14, C.cyan, {bold:true});
  text(slide, heading, 64, 68, 1130, 66, 34, C.white, {bold:true});
  if (sub) text(slide, sub, 66, 138, 1100, 42, 18, C.muted);
}
function footer(slide, n, owner) {
  text(slide, `${String(n).padStart(2,'0')}  /  OpenSearch`, 64, 682, 250, 20, 11, C.muted, {bold:true});
  text(slide, owner, 980, 682, 235, 20, 11, C.muted, {bold:true});
}
function base(kicker, heading, sub, owner) {
  const s = ppt.slides.add(); s.background.fill = C.navy; title(s,kicker,heading,sub); footer(s,ppt.slides.items.length,owner); return s;
}
function note(slide, body) { slide.speakerNotes.textFrame.setText(body); }
function arrow(slide,x1,y1,x2,y2,color=C.cyan) { slide.shapes.add({ geometry:'line', position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1)}, fill:'none', line:{fill:color,width:3,beginArrowType:'none',endArrowType:'triangle'} }); }
function bullet(slide, label, body, x,y,w, color=C.cyan) { rect(slide,x,y,8,8,color,4); text(slide,label,x+20,y-7,w,27,20,C.white,{bold:true}); if(body) text(slide,body,x+20,y+20,w,35,15,C.muted); }

// 1 Cover
{
 const s=ppt.slides.add(); s.background.fill=C.navy; s.images.add({blob:imgBytes,contentType:'image/png',alt:'OpenSearch architecture illustration',fit:'cover',position:{left:0,top:0,width:W,height:H}});
 rect(s,0,0,690,H,{color:C.navy,transparency:18},0,'none');
 text(s,'TEAM 3  ·  15–30 MIN',64,56,460,24,16,C.cyan,{bold:true});
 text(s,'How OpenSearch\nworks',64,118,560,150,48,C.white,{bold:true});
 text(s,'From document ingestion to distributed search results',68,300,470,55,22,C.muted);
 text(s,'Operating principles, data flow, and scaling',68,366,460,35,17,C.white,{bold:true});
 text(s,'Speaker split: 1) foundations  2) indexing & search  3) scale & applications',68,622,560,30,14,C.muted);
 note(s,'Opening: OpenSearch is a distributed search and analytics engine built on Apache Lucene. In this talk we follow one document and one query through the system. Source: https://docs.opensearch.org/latest/getting-started/ and https://opensearch.org/About/');
}

// 2 mental model
{
 const s=base('01 · Big picture','OpenSearch is a pipeline, not just a search box','Data enters, becomes searchable structure, then returns ranked results.','Speaker 1');
 const xs=[90,360,630,900], labels=['Sources','Ingest','OpenSearch core','Use']; const subs=['logs · apps · files','Data Prepper\npipelines','index · shard ·\nLucene','REST API ·\nDashboards'];
 xs.forEach((x,i)=>{rect(s,x,250,200,130,i===2?C.blue:C.navy2,18,C.line); text(s,labels[i],x+18,272,164,32,24,C.white,{bold:true}); text(s,subs[i],x+18,315,164,48,16,C.muted); if(i<3) arrow(s,x+205,315,xs[i+1]-10,315,i===1?C.amber:C.cyan);});
 text(s,'Core idea',90,470,160,25,15,C.amber,{bold:true}); text(s,'OpenSearch separates the user-facing interfaces from the distributed engine that stores and searches data.',90,500,1050,54,24,C.white,{bold:true});
 note(s,'Explain the whole-system view first. OpenSearch documentation names Data Prepper, OpenSearch core, Dashboards, and clients as the main components. Source: https://docs.opensearch.org/latest/getting-started/');
}

// 3 components
{
 const s=base('02 · Components','Three layers work together','Each layer has a distinct responsibility.','Speaker 1');
 const cards=[['Data Prepper','Collects, filters, enriches, and routes data before indexing.','INGEST',C.cyan],['OpenSearch','Stores JSON documents, builds indexes, executes queries and aggregations.','SEARCH',C.blue],['Dashboards + clients','REST API, language clients, Discover, Visualize, and dashboards.','USE',C.amber]];
 cards.forEach((c,i)=>{const x=70+i*390; rect(s,x,228,340,260,C.navy2,20,C.line); text(s,c[2],x+24,252,100,22,13,c[3],{bold:true}); text(s,c[0],x+24,292,292,40,26,C.white,{bold:true}); text(s,c[1],x+24,354,284,92,18,C.muted);});
 text(s,'A typical request path',70,560,260,25,15,C.cyan,{bold:true}); text(s,'client → coordinating node → target shards → merged response',70,592,950,32,23,C.white,{bold:true});
 note(s,'Stress that OpenSearch is a suite. Data Prepper is an external collector, ingest pipelines run inside the cluster, and Dashboards is the visualization UI. Sources: https://docs.opensearch.org/latest/getting-started/ and https://docs.opensearch.org/latest/ingest-pipelines/');
}

// 4 objects
{
 const s=base('03 · Data model','The mental model: JSON → index → shards','These objects explain where data lives and how OpenSearch searches it.','Speaker 1');
 const items=[['Document','One JSON record'],['Index','Collection of related documents'],['Mapping','Field types and indexing rules'],['Shard','A subset of an index'],['Replica','Copy for resilience and search capacity']];
 items.forEach((it,i)=>{const x=80+(i%3)*390,y=230+Math.floor(i/3)*160; rect(s,x,y,330,112,i===3?C.blue:C.navy2,16,C.line); text(s,it[0],x+18,y+18,285,29,22,C.white,{bold:true}); text(s,it[1],x+18,y+57,285,25,16,C.muted);});
 text(s,'Rule of thumb',80,570,150,22,14,C.amber,{bold:true}); text(s,'An index is logical; shards make it physically distributable.',80,600,900,30,22,C.white,{bold:true});
 note(s,'Define the terms before showing the mechanics. The official concepts page describes documents as JSON, indexes as collections, mappings as schemas, and shards/replicas as the distributed storage units. Source: https://docs.opensearch.org/latest/getting-started/concepts/');
}

// 5 ingest/index
{
 const s=base('04 · Indexing','Indexing turns text into searchable structures','The analyzer and inverted index are the key ideas.','Speaker 2');
 const steps=[['1','JSON arrives','{ "title": "RAG systems" }'],['2','Analyze','lowercase · tokenize · normalize'],['3','Build index','term → document IDs + positions'],['4','Persist','Lucene segments inside a shard']];
 steps.forEach((st,i)=>{const x=72+i*292; rect(s,x,232,248,220,i===2?C.blue:C.navy2,18,C.line); text(s,st[0],x+20,252,30,30,20,C.amber,{bold:true}); text(s,st[1],x+20,302,208,30,21,C.white,{bold:true}); text(s,st[2],x+20,354,202,66,16,C.muted); if(i<3) arrow(s,x+250,342,x+280,342,C.cyan);});
 text(s,'Inverted index',72,520,180,26,15,C.cyan,{bold:true}); text(s,'retrieval  →  doc 1, doc 4, doc 9',72,556,480,30,24,C.white,{bold:true}); text(s,'The index stores terms and their document locations so search does not scan every document from scratch.',610,535,530,70,20,C.muted);
 note(s,'Explain that text analysis happens at index time and query time. The inverted index maps words to the documents containing them, and positions support phrase queries. Source: https://docs.opensearch.org/latest/getting-started/intro/');
}

// 6 query flow
{
 const s=base('05 · Search','A query fans out, then results fan in','The coordinating node routes work and merges shard-level answers.','Speaker 2');
 rect(s,74,255,180,110,C.navy2,18,C.line); text(s,'Client query',94,282,140,28,22,C.white,{bold:true}); text(s,'GET /reviews/_search',94,320,145,22,14,C.cyan);
 arrow(s,260,310,370,310,C.amber);
 rect(s,390,255,200,110,C.blue,18,'none'); text(s,'Coordinating\nnode',416,282,150,55,22,C.white,{bold:true});
 const ys=[200,310,420]; ys.forEach((y,i)=>{arrow(s,602,310,680,y,C.cyan); rect(s,700,y-42,180,84,C.navy2,16,C.line); text(s,`Shard ${i+1}`,730,y-12,115,24,20,C.white,{bold:true}); arrow(s,892,y,1010,310,C.cyan);});
 rect(s,1030,255,180,110,C.amber,18,'none'); text(s,'Merged\nresponse',1054,282,138,55,22,C.navy,{bold:true});
 text(s,'Each shard returns local hits and scores. The coordinator combines them into one ranked response.',90,565,1060,40,21,C.muted);
 note(s,'This is the most important operational flow. The request may arrive at any node, which coordinates shard execution, gathers results, and returns the final response. Source: https://docs.opensearch.org/latest/getting-started/intro/ and https://docs.opensearch.org/latest/search-plugins/searching-data/search-shard-routing/');
}

// 7 query DSL and relevance
{
 const s=base('06 · Querying','Query DSL expresses intent; BM25 ranks relevance','A short JSON query can combine full-text search, filters, and aggregations.','Speaker 2');
 rect(s,72,226,500,320,'#091D3C',18,C.line); text(s,'GET /docs/_search',96,250,440,28,18,C.cyan,{bold:true}); text(s,'{\n  "query": {\n    "match": { "body": "retrieval augmented generation" }\n  }\n}',96,300,420,180,20,C.white,{margin:4});
 rect(s,650,236,480,90,C.navy2,16,C.line); text(s,'Analyzer',678,258,180,25,18,C.cyan,{bold:true}); text(s,'uses the same field analysis rules',678,293,390,25,17,C.muted);
 rect(s,650,356,480,90,C.navy2,16,C.line); text(s,'BM25',678,378,180,25,18,C.amber,{bold:true}); text(s,'scores term frequency and rarity',678,413,390,25,17,C.muted);
 rect(s,650,476,480,90,C.navy2,16,C.line); text(s,'Response',678,498,180,25,18,C.green,{bold:true}); text(s,'hits, score, fields, aggregations',678,533,390,25,17,C.muted);
 note(s,'Show that Query DSL is JSON-based. For analyzed text, the query is processed with the field’s analyzer; BM25 is used for relevance scoring. Source: https://docs.opensearch.org/latest/query-dsl/ and https://docs.opensearch.org/latest/getting-started/intro/');
}

// 8 scale
{
 const s=base('07 · Scale','Shards distribute data; replicas protect and accelerate','Scaling is a trade-off: more shards are not automatically better.','Speaker 3');
 const nodes=[['Node A','Primary 0','Replica 1'],['Node B','Primary 1','Replica 0'],['Node C','Replica 2','Primary 2']];
 nodes.forEach((n,i)=>{const x=82+i*382; rect(s,x,245,300,190,C.navy2,18,C.line); text(s,n[0],x+20,265,260,28,22,C.white,{bold:true}); rect(s,x+20,318,110,58,i===1?C.amber:C.blue,12,'none'); text(s,n[1],x+32,334,90,22,15,C.white,{bold:true}); rect(s,x+160,318,110,58,C.navy,12,C.line); text(s,n[2],x+172,334,90,22,15,C.muted,{bold:true});});
 text(s,'Benefits',82,516,120,25,15,C.cyan,{bold:true}); text(s,'distribution  ·  failover  ·  parallel search',82,552,700,30,24,C.white,{bold:true});
 text(s,'Caution',82,610,120,25,15,C.amber,{bold:true}); text(s,'Every shard is a Lucene index and consumes CPU + memory.',82,642,900,25,18,C.muted);
 note(s,'Use the diagram to explain primary versus replica. Replicas can serve searches and take over after failure. The docs caution that too many shards strain a cluster; shard sizing and workload need benchmarking. Sources: https://docs.opensearch.org/latest/getting-started/intro/ and https://docs.opensearch.org/latest/tuning-your-cluster/');
}

// 9 ingest pipeline + vector
{
 const s=base('08 · Modern search','Keyword search and vector search can coexist','OpenSearch supports classic text retrieval, vector search, and hybrid applications.','Speaker 3');
 const cols=[['Classic','BM25\nExact terms + relevance',C.cyan],['Vector','Embeddings\nSemantic similarity',C.blue],['Hybrid','Combine scores\nBetter recall + intent',C.amber]];
 cols.forEach((c,i)=>{const x=90+i*380; rect(s,x,248,310,180,C.navy2,20,C.line); text(s,c[0],x+24,274,260,30,24,c[2],{bold:true}); text(s,c[1],x+24,330,260,56,19,C.white,{bold:true}); if(i<2) text(s,'+',x+326,320,35,35,30,C.muted,{bold:true});});
 text(s,'Example RAG flow',90,500,220,25,15,C.cyan,{bold:true}); text(s,'embed query → retrieve passages → generate answer with citations',90,536,1020,35,24,C.white,{bold:true}); text(s,'Keep this as an extension: the core search principles remain indexing, routing, scoring, and retrieval.',90,590,1040,35,17,C.muted);
 note(s,'Present vector and hybrid search as a modern extension, not as a replacement for the core engine. OpenSearch positions itself as a data store and vector database for AI applications. Source: https://docs.opensearch.org/latest/getting-started/');
}

// 10 dashboard/security
{
 const s=base('09 · Operations','From raw events to an operational dashboard','A useful deployment makes search observable, secure, and easy to inspect.','Speaker 3');
 const steps=[['Collect','logs / traces / app events'],['Transform','pipeline processors'],['Index','mapping + shards'],['Explore','Discover / Query DSL'],['Monitor','visualizations + alerts']];
 steps.forEach((st,i)=>{const x=58+i*240; rect(s,x,270,202,128,i===4?C.amber:C.navy2,16,C.line); text(s,st[0],x+18,292,165,28,20,i===4?C.navy:C.white,{bold:true}); text(s,st[1],x+18,338,166,40,15,i===4?C.navy:C.muted); if(i<4) arrow(s,x+206,334,x+231,334,C.cyan);});
 bullet(s,'Security', 'authentication, authorization, encrypted transport, audit controls',80,486,1000,C.green);
 bullet(s,'Demo idea', 'open Dashboards → search a field → filter time range → build one chart',80,566,1000,C.cyan);
 note(s,'Connect architecture to an operator’s experience. Dashboards provides Discover, Visualize, Dashboards, and Dev Tools. Security is part of production readiness, but keep the details brief for this talk. Source: https://docs.opensearch.org/latest/dashboards/getting-started/learn-dashboards/');
}

// 11 conclusion
{
 const s=base('10 · Takeaway','One sentence to remember','OpenSearch makes large collections searchable by turning documents into distributed, analyzed, and ranked data.','All speakers');
 const big='INGEST  →  INDEX  →  ROUTE  →  SCORE  →  VISUALIZE'; text(s,big,95,245,1080,54,31,C.white,{bold:true});
 const qs=[['Why fast?','Inverted indexes avoid full scans.'],['Why scalable?','Shards distribute work; replicas add resilience.'],['Why useful?','Query DSL, APIs, and Dashboards turn results into action.']];
 qs.forEach((q,i)=>{const y=360+i*78; text(s,q[0],104,y,190,25,18,i===0?C.cyan:i===1?C.amber:C.green,{bold:true}); text(s,q[1],320,y,740,25,19,C.muted);});
 text(s,'Q&A',104,620,140,30,24,C.cyan,{bold:true}); text(s,'Thank you',925,620,220,30,24,C.white,{bold:true});
 note(s,'Closing: repeat the five verbs and invite questions. Sources used throughout: https://docs.opensearch.org/latest/getting-started/, https://docs.opensearch.org/latest/getting-started/concepts/, https://docs.opensearch.org/latest/query-dsl/, https://docs.opensearch.org/latest/search-plugins/searching-data/search-shard-routing/');
}

const candidate = path.join(TMP_DIR,'candidate.pptx');
await (await PresentationFile.exportPptx(ppt)).save(candidate);
console.log(candidate);
