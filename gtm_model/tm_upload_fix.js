// Teachable Machine (Audio Project) fixes for large uploads on Windows – paste ONE line in F12 -> Console, press Enter.
//  1) Upload fix: Windows Chrome labels .zip files "application/x-zip-compressed"; Teachable Machine only accepts
//     "application/zip" ("File Input Error: You can only upload zip files ..."). Every upload box (also classes added
//     later) is fixed once per second.
//  2) Training fix: with thousands of samples Teachable Machine crashes in loadExamples with
//     "RangeError: Maximum call stack size exceeded" (String.fromCharCode.apply on a huge array).
//     The line makes that call work in chunks.
//  3) If a crashed training left the button stuck on "Training...", it is reset so you can click Train Model again.
// Works until the page is reloaded (then paste it again). First time only: type  allow pasting  and press Enter.
(function(){var o=String.fromCharCode;if(!o.apply.__ss){var a=function(t,x){var s='';for(var i=0;i<x.length;i+=8192)s+=Function.prototype.apply.call(o,t,Array.prototype.slice.call(x,i,i+8192));return s};a.__ss=1;o.apply=a}function d(r,s,z=[]){r.querySelectorAll('*').forEach(e=>{if(e.tagName===s)z.push(e);if(e.shadowRoot)d(e.shadowRoot,s,z)});return z}function f(){let n=0;d(document,'TM-FILE-SAMPLE-INPUT').forEach(e=>{if(String(e.accept).includes('zip')&&!String(e.accept).includes('x-zip')){e.accept+=', application/x-zip-compressed, application/x-zip, application/octet-stream';n++}});if(n)console.log('SonicSentinel: '+n+' new upload boxes fixed')}d(document,'TM-TRAIN').forEach(t=>{if(t.state==='pretraining'||t.state==='training'){t.state='waiting'}});clearInterval(window.__ssFix);f();window.__ssFix=setInterval(f,1000);console.log('SonicSentinel upload + training fix is ON')})();
