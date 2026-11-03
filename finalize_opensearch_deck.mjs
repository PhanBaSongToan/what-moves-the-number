import path from 'node:path';
import { pathToFileURL } from 'node:url';
const SKILL_DIR='C:/Users/ACER/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const workspaceDir='C:/Users/ACER/Code/TMG301';
const candidatePath=path.join(workspaceDir,'.opensearch-build','candidate.pptx');
const finalPath=path.join(workspaceDir,'outputs','OpenSearch_Operating_Principles_Team3_final.pptx');
const { finalizePresentation }=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
  workspaceDir,candidatePath,finalPath,
  pythonExecutable:'C:/Users/ACER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  requiredNativeTableOwnerSlides:[],
  fontPolicy:{basis:'design',families:['Aptos']},
  verifyArtifactToolImport:true,
  receiptPath:path.join(workspaceDir,'.opensearch-build','OpenSearch_Operating_Principles_Team3_final.validation.json'),
});
console.log(JSON.stringify(result,null,2));
