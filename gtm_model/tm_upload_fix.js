// Teachable Machine (Audio Project) – allow our _tm_upload/*.zip files on Windows.
// Windows Chrome reports .zip files as "application/x-zip-compressed", but Teachable Machine only accepts
// the type "application/zip" and shows "File Input Error: You can only upload zip files ...".
// Usage: F12 -> Console, type  allow pasting  + Enter (first time only), paste the line below, Enter.
// It keeps fixing every upload box once per second, also for classes you add later (until the page is reloaded).
(function(){function d(r,a=[]){r.querySelectorAll('*').forEach(e=>{if(e.tagName==='TM-FILE-SAMPLE-INPUT')a.push(e);if(e.shadowRoot)d(e.shadowRoot,a)});return a}function f(){let n=0;d(document).forEach(e=>{if(String(e.accept).includes('zip')&&!String(e.accept).includes('x-zip')){e.accept+=', application/x-zip-compressed, application/x-zip, application/octet-stream';n++}});if(n)console.log('SonicSentinel: '+n+' new upload boxes fixed')}clearInterval(window.__ssFix);f();window.__ssFix=setInterval(f,1000);console.log('SonicSentinel upload fix is ON')})();
