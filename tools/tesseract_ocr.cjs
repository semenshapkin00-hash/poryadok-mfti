// Local-only bounded OCR worker. No URL input and no runtime weight downloads.
const fs = require('node:fs');
const path = require('node:path');
const modulePath=process.env.NAVIGATOR_TESSERACT_MODULE;
if(!modulePath || !path.isAbsolute(modulePath)) throw new Error('Local tesseract.js package path required');
const {createWorker}=require(modulePath);
(async()=>{
  const manifest=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
  if(!path.isAbsolute(manifest.langPath) || /^(?:https?|file):/i.test(manifest.langPath)) throw new Error('Only local language weights permitted');
  for(const lang of ['rus','eng']) if(!fs.statSync(path.join(manifest.langPath,lang+'.traineddata')).isFile()) throw new Error('Local weights unavailable');
  if(!Array.isArray(manifest.images) || manifest.images.length>20) throw new Error('OCR page limit exceeded');
  for(const image of manifest.images) if(!path.isAbsolute(image.path) || !fs.statSync(image.path).isFile()) throw new Error('Only local rendered images permitted');
  const worker=await createWorker('rus+eng',1,{langPath:manifest.langPath,cachePath:manifest.cachePath,gzip:false,cacheMethod:'none',logger:()=>{}});
  try{
    await worker.setParameters({tessedit_pageseg_mode:'6',preserve_interword_spaces:'1'});
    const pages=[];let outputBytes=0;
    for(const image of manifest.images){
      const {data}=await worker.recognize(image.path,{}, {text:true,tsv:true});
      const lines=data.tsv.trim().split('\n').slice(1).map(row=>row.split('\t')).filter(v=>v[0]==='5'&&v.length>=12&&v.slice(11).join('\t').trim()).map(v=>({
        text:v.slice(11).join('\t'),confidence:Math.max(0,Math.min(100,Number(v[10])))/100,
        block:Number(v[2]),paragraph:Number(v[3]),line:Number(v[4]),word:Number(v[5]),
        bbox:[Number(v[6])/(image.scale_x||image.scale),Number(v[7])/(image.scale_y||image.scale),(Number(v[6])+Number(v[8]))/(image.scale_x||image.scale),(Number(v[7])+Number(v[9]))/(image.scale_y||image.scale)]
      }));
      if(lines.length>30000) throw new Error('OCR word limit exceeded');
      const page={page:image.page,width:image.width,height:image.height,coordinate_system:image.coordinate_system,lines,text:data.text};
      outputBytes+=Buffer.byteLength(JSON.stringify(page),'utf8');
      if(outputBytes>10*1024*1024) throw new Error('OCR output limit exceeded');
      pages.push(page);
    }
    process.stdout.write(JSON.stringify({engine:'Tesseract.js 7 / WASM / rus+eng',pages}));
  }finally{await worker.terminate();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});
