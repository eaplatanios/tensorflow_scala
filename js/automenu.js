// jQuery(document).ready(function() {
//     activeLinks();
//     activeToggle();
//     organizeContent();
// });
//
// function organizeContent() {
//     var content = $('#content');
//     var subcontent = $('<div></div>');
//     content.prepend(subcontent);
//     content.find('h1').each(function (index) {
//         var section = $('<section></section>');
//         subcontent.append(section);
//         var h1 = $(this);
//         var elements = h1.nextUntil('h1');
//         var text = h1.text();
//         var slug = slugify(text) + '-' + index;
//         addSectionToSidebar(text, slug);
//         section.append(makeSectionAnchor(h1, text, slug));
//         if (elements.length > 0) {
//             elements.appendTo(section);
//             organizeSubSection(slug, elements);
//         }
//     });
//     removeEmptyList();
// }
//
// function organizeSubSection(s, children) {
//     children.filter('h2').each(function (index, el) {
//         var h2 = $(this);
//         var text = h2.text();
//         var slug = s + '-' + slugify(text) + '-' + index;
//         var a = makeSectionAnchor(h2, text, slug);
//         addSubSectionToSidebar(text, slug, s);
//     });
// }
//
// function makeSectionAnchor(h, text, slug) {
//     var a = $('<a></a>').attr({
//         'class': 'anchor',
//         'name': slug,
//         'href': '#' + slug
//     });
//     a.append(h.clone());
//     h.replaceWith(a);
//     return a;
// }
//
// function addSectionToSidebar(text, slug) {
//     var ul = $('<ul></ul>').addClass('sub_section');
//     var a = $('<a href="#' + slug + '">' + text + '<span><i class="fa fa-angle-right"></i></span></a>');
//     a.find('.fa-angle-right').css('padding-top', '0.7em');
//     var li = $('<li class="' + slug + '"></li>');
//     li.append(a).append(ul);
//     ul.hide();
//     var sidebar = $('#sidebar');
//     sidebar.append(li);
//     a.click(function (event) {
//         sidebar.find('li').add('#sidebar a').removeClass('active');
//         sidebar.find('.sub_section').not(ul).slideUp();
//         ul.slideToggle('fast');
//         li.add(a).toggleClass('active');
//     });
// }
//
// function addSubSectionToSidebar(text, slug, s) {
//     var ul = $('#sidebar').find('li.' + s + ' ul');
//     var li = $('<li class="' + slug + '"><a href="#' + slug + '">' + text + '</a></li>');
//     ul.append(li);
// }
//
// function removeEmptyList() {
//     $('#sidebar').find('>li').not('.sidebar-brand').each(function (index, el) {
//         var li = $(this);
//         var children = li.find('li');
//         if (children.size() === 0) {
//             li.find('span').remove();
//         }
//     });
// }
//
// function slugify(text) {
//     return text.toString().toLowerCase()
//         .replace(/\s+/g, '-') // Replace spaces with -
//         .replace(/[^\w\-]+/g, '') // Remove all non-word chars
//         .replace(/\-\-+/g, '-') // Replace multiple - with single -
//         .replace(/^-+/, '') // Trim - from start of text
//         .replace(/-+$/, ''); // Trim - from end of text
// }
//
// function activeToggle() {
//     $("#menu-toggle").click(function (e) {
//         e.preventDefault();
//         $("#wrapper").toggleClass("toggled");
//     });
// }
//
// function activeLinks() {
//     $('a[data-href]').each(function (index, el) {
//         $(this).attr('href', $(this).attr('data-href'));
//     });
// }