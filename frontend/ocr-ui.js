// Pure presentation helpers: no inference, DOM access, or network requests.
export function supportedUploadFormats(health={}){
  const base=['txt','md','pdf','docx','xlsx'];
  const known=[...base,'png','jpg','jpeg'];
  const advertised=Array.isArray(health.supported_formats)?health.supported_formats.map(value=>String(value).trim().toLowerCase().replace(/^\./,'')):base;
  const imagesEnabled=health.models?.ocr?.available===true;
  const formats=known.filter(value=>advertised.includes(value)&&(imagesEnabled||base.includes(value)));
  const usable=formats.length?formats:base;
  const labels={txt:'TXT',md:'Markdown',pdf:'PDF',docx:'DOCX',xlsx:'XLSX',png:'PNG',jpg:'JPEG',jpeg:'JPEG'};
  return{formats:usable,accept:usable.map(value=>'.'+value).join(','),label:[...new Set(usable.map(value=>labels[value]))].join(', '),imagesEnabled};
}

export function sourceExtraction(doc={},citation=null){
  const clauses=Array.isArray(doc.clauses)?doc.clauses:[];
  const clause=citation?clauses.find(value=>value.id===citation.clause_id)||clauses.find(value=>value.text===citation.quote):null;
  const evidence=citation?.evidence||clause?.evidence||{};
  const extraction=doc.extraction||{};
  const page=Number(citation?.page||clause?.page||1);
  const ocrPages=Array.isArray(extraction.ocr_pages)?extraction.ocr_pages:[];
  const unreadablePages=Array.isArray(extraction.unreadable_pages)?extraction.unreadable_pages:[];
  const explicit=citation?.provenance||clause?.provenance||evidence.provenance;
  const method=extraction.method||doc.metadata?.extraction_provenance||'';
  const hasOCR=ocrPages.length>0||['ocr','pdf_text+ocr'].includes(method)||clauses.some(value=>['ocr','pdf_text+ocr'].includes(value.provenance));
  const isOCR=citation?(explicit?['ocr','pdf_text+ocr'].includes(explicit):ocrPages.includes(page)||method==='ocr'):hasOCR;
  const raw=citation?.ocr_confidence??clause?.ocr_confidence??evidence.ocr_confidence;
  const score=raw!=null&&Number.isFinite(Number(raw))&&Number(raw)>=0&&Number(raw)<=1?Number(raw):null;
  const words=citation?.word_confidences||clause?.word_confidences||evidence.word_confidences||[];
  const review=Boolean(citation?.requires_review||clause?.requires_review||evidence.requires_review||isOCR||extraction.requires_review||['review','needs_review','unreadable'].includes(doc.status));
  return{isOCR,hasOCR,review,score,ocrPages,unreadablePages,
    numeric:Boolean(citation?.numeric_uncertainty||clause?.numeric_uncertainty||evidence.numeric_uncertainty),
    words:Array.isArray(words)?words:[],
    provenance:explicit||method,
    coordinateSystem:citation?.coordinate_system||clause?.coordinate_system||evidence.coordinate_system||null};
}

export function recognitionLabel(score){
  return score==null||!Number.isFinite(Number(score))||Number(score)<0||Number(score)>1?'Оценка распознавания не передана':`Оценка распознавания: ${Math.round(score*100)} из 100`;
}

export function previewRectangle(box,width,height){
  if(!Array.isArray(box)||box.length!==4||!box.every(value=>Number.isFinite(Number(value)))||!(width>0)||!(height>0))return null;
  const[x0,y0,x1,y1]=box.map(Number);
  if(x0<0||y0<0||x1<=x0||y1<=y0||x1>width+1||y1>height+1)return null;
  return{left:x0/width*100,top:y0/height*100,width:Math.min(100,x1/width*100)-x0/width*100,height:Math.min(100,y1/height*100)-y0/height*100};
}
