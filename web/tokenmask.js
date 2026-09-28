// From the connector token study (2026-09-24): logo isolation and contour tracing.
// Browser-only logo isolation. No service calls or machine-learning dependencies.
export function makeMask(image,{mode='auto',threshold=110,invert=false,size=384}={}) {
 const c=document.createElement('canvas');c.width=c.height=size;const g=c.getContext('2d',{willReadFrequently:true});
 const scale=(size-12)/Math.max(image.width,image.height),w=image.width*scale,h=image.height*scale,x=(size-w)/2,y=(size-h)/2;
 g.drawImage(image,x,y,w,h);const rgba=g.getImageData(0,0,size,size).data;
 const inside=(i)=>{const px=i%size,py=Math.floor(i/size);return px>=Math.ceil(x)&&px<Math.floor(x+w)&&py>=Math.ceil(y)&&py<Math.floor(y+h)};
 let transparent=false;for(let i=0;i<size*size;i++)if(inside(i)&&rgba[i*4+3]<240){transparent=true;break;}
 if(mode==='auto')mode=transparent?'alpha':'background';
 // Median of perimeter samples is more robust than a single corner pixel.
 const channels=[[],[],[]];for(let i=0;i<size*size;i++)if(inside(i)){const px=i%size,py=Math.floor(i/size);if(px<=Math.ceil(x)+1||px>=Math.floor(x+w)-2||py<=Math.ceil(y)+1||py>=Math.floor(y+h)-2)for(let k=0;k<3;k++)channels[k].push(rgba[i*4+k]);}
 const bg=channels.map(a=>a.sort((a,b)=>a-b)[Math.floor(a.length/2)]??255);
 let bits=new Uint8Array(size*size);
 for(let i=0;i<bits.length;i++){if(!inside(i))continue;const j=i*4,alpha=rgba[j+3],lum=.299*rgba[j]+.587*rgba[j+1]+.114*rgba[j+2];let selected;
 if(mode==='alpha')selected=alpha>threshold;else if(mode==='dark')selected=lum<threshold&&alpha>127;else if(mode==='light')selected=lum>threshold&&alpha>127;else selected=Math.hypot(rgba[j]-bg[0],rgba[j+1]-bg[1],rgba[j+2]-bg[2])>threshold&&alpha>127;
 bits[i]=invert?!selected:+selected;}
 // Trim empty space and fit the mark consistently onto the token.
 let minX=size,minY=size,maxX=-1,maxY=-1;bits.forEach((v,i)=>{if(v){minX=Math.min(minX,i%size);maxX=Math.max(maxX,i%size);minY=Math.min(minY,Math.floor(i/size));maxY=Math.max(maxY,Math.floor(i/size));}});
 const temp=document.createElement('canvas');temp.width=temp.height=size;const tg=temp.getContext('2d');const pixels=tg.createImageData(size,size);bits.forEach((v,i)=>{pixels.data.set([v*255,v*255,v*255,255],i*4)});tg.putImageData(pixels,0,0);
 g.fillStyle='black';g.fillRect(0,0,size,size);
 if(maxX>=0){const sw=maxX-minX+1,sh=maxY-minY+1,s=(size-16)/Math.max(sw,sh);g.drawImage(temp,minX,minY,sw,sh,(size-sw*s)/2,(size-sh*s)/2,sw*s,sh*s);}
 const out=g.getImageData(0,0,size,size);bits=new Uint8Array(size*size);for(let i=0;i<bits.length;i++){bits[i]=out.data[i*4]>127?1:0;out.data.set([bits[i]*255,bits[i]*255,bits[i]*255,255],i*4)}g.putImageData(out,0,0);
 const result={canvas:c,bits,size,mode,empty:maxX<0};
 result.loops=contours(result).map(smoothOutline);
 // Rasterize the exact geometry contours at double resolution. Canvas coverage
 // supplies gray edge pixels; holes use the same even-odd fill as the extrusion.
 const aa=document.createElement('canvas');aa.width=aa.height=size*2;const ag=aa.getContext('2d');
 ag.fillStyle='black';ag.fillRect(0,0,aa.width,aa.height);ag.scale(2,2);ag.beginPath();
 for(const loop of result.loops){loop.forEach(([x,y],i)=>i?ag.lineTo(x,y):ag.moveTo(x,y));ag.closePath();}
 ag.fillStyle='white';ag.fill('evenodd');result.canvas=aa;
 return result;
}
// Trace cell boundaries into closed loops; preserve internal holes and disconnected marks.
export function contours({bits,size,loops:cached}) {
 if(cached)return cached;
 const edges=new Map(),key=(x,y)=>y*(size+1)+x;
 const add=(x,y,a,b)=>{const k=key(x,y);if(!edges.has(k))edges.set(k,[]);edges.get(k).push(key(a,b));};
 const on=(x,y)=>x>=0&&y>=0&&x<size&&y<size&&bits[y*size+x];
 for(let y=0;y<size;y++)for(let x=0;x<size;x++)if(on(x,y)){if(!on(x,y-1))add(x,y,x+1,y);if(!on(x+1,y))add(x+1,y,x+1,y+1);if(!on(x,y+1))add(x+1,y+1,x,y+1);if(!on(x-1,y))add(x,y+1,x,y);}
 const loops=[];while(edges.size){const start=edges.keys().next().value;let at=start,loop=[],guard=0;do{loop.push([at%(size+1),Math.floor(at/(size+1))]);const next=edges.get(at);if(!next)break;const to=next.pop();if(!next.length)edges.delete(at);at=to;}while(at!==start&&++guard<size*size*4);if(at===start&&loop.length>=4)loops.push(loop.filter((p,i)=>{const a=loop[(i+loop.length-1)%loop.length],b=loop[(i+1)%loop.length];return (p[0]-a[0])*(b[1]-p[1])!==(p[1]-a[1])*(b[0]-p[0]);}));}
 return loops;
}

// Subpixel simplification removes the stair steps before bounded corner rounding.
function smoothOutline(loop){
 if(loop.length<8)return loop;
 const distance=(p,a,b)=>{const dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy||1)));return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);};
 function simplify(points){if(points.length<3)return points;let far=.65,index=-1;for(let i=1;i<points.length-1;i++){const d=distance(points[i],points[0],points.at(-1));if(d>far){far=d;index=i;}}return index<0?[points[0],points.at(-1)]:[...simplify(points.slice(0,index+1)).slice(0,-1),...simplify(points.slice(index))];}
 const split=Math.floor(loop.length/2);let points=[...simplify(loop.slice(0,split+1)).slice(0,-1),...simplify([...loop.slice(split),loop[0]]).slice(0,-1)];
 // Only round within 0.65 source pixels, preserving intentional large corners.
 const rounded=[];for(let i=0;i<points.length;i++){const p=points[i],prev=points[(i+points.length-1)%points.length],next=points[(i+1)%points.length];const inLen=Math.hypot(p[0]-prev[0],p[1]-prev[1]),outLen=Math.hypot(next[0]-p[0],next[1]-p[1]);const f=Math.min(.2,.65/(inLen||1)),g=Math.min(.2,.65/(outLen||1));const a=[p[0]+(prev[0]-p[0])*f,p[1]+(prev[1]-p[1])*f],b=[p[0]+(next[0]-p[0])*g,p[1]+(next[1]-p[1])*g];for(const t of [0,.5,1])rounded.push([(1-t)*(1-t)*a[0]+2*(1-t)*t*p[0]+t*t*b[0],(1-t)*(1-t)*a[1]+2*(1-t)*t*p[1]+t*t*b[1]]);}
 return rounded;
}
