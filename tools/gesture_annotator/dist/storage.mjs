const DATABASE = 'kfcore-gesture-annotator-v1';
let database;
function open() {
  database ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('projects');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('本地数据库被其他页面阻塞，请关闭旧页面'));
  });
  return database;
}
export async function loadProject(hash) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const request = db.transaction('projects').objectStore('projects').get(hash);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function saveProject(project) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('projects', 'readwrite');
    tx.objectStore('projects').put(project, project.video.sha256);
    tx.oncomplete = resolve;
    tx.onabort = () => reject(tx.error ?? new Error('保存事务取消'));
    tx.onerror = () => reject(tx.error);
  });
}
