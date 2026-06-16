// Handle sending current preview to parent and returning
document.addEventListener('DOMContentLoaded', function () {
  function safePostMessage(obj) {
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage(obj, "*");
        console.debug('Posted message to parent', obj);
      } else {
        console.debug('No parent window found for postMessage', obj);
      }
    } catch (e) {
      console.error('postMessage failed', e);
    }
  }

  const postBtn = document.getElementById('post-to-parent-btn');
  if (postBtn) {
    postBtn.addEventListener('click', function () {
      // collect some fields from the page to send as preview
      const rawJsonEl = document.querySelector('.raw-json');
      let payload = { type: 'cnc_preview', time: new Date().toISOString() };
      if (rawJsonEl) {
        payload.data = rawJsonEl.innerText || rawJsonEl.textContent || null;
      }
      safePostMessage(payload);
      postBtn.classList.add('btn-sent');
    });
  }

  const returnBtn = document.getElementById('return-btn');
  if (returnBtn) {
    returnBtn.addEventListener('click', function () {
      safePostMessage({ type: 'cnc_return', time: new Date().toISOString() });
    });
  }
});
